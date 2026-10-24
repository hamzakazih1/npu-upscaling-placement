from __future__ import annotations

from padmidi.modes.base import Field, Mode, clamp01
from padmidi.scales import note_name
from padmidi.touch import Contact


class TriggerMode(Mode):
    """A grid of drum / clip-launch pads. Notes rise left to right, then
    bottom to top, starting at ``base_note`` (C1 = a drum rack's kick)."""

    key = "trigger"
    title = "Trigger"
    defaults = {
        "channel": 1,
        "rows": 4,
        "cols": 6,
        "base_note": 36,
        "velocity": "pressure",
        "fixed_velocity": 100,
        "velocity_curve": 100,
        "aftertouch": False,
        "onset_frames": 2,
    }
    fields = [
        Field("channel", "MIDI channel", "int", 1, 16),
        Field("rows", "Rows", "int", 1, 8),
        Field("cols", "Columns", "int", 1, 8),
        Field("base_note", "First note (bottom-left)", "note", 0, 127),
        Field("velocity", "Velocity from", "choice", options=["pressure", "position", "fixed"],
              help="pressure: how firmly you tap; position: higher in the pad is louder"),
        Field("fixed_velocity", "Fixed velocity", "int", 1, 127),
        Field("velocity_curve", "Velocity curve % (100 = linear)", "int", 25, 400),
        Field("aftertouch", "Polyphonic aftertouch from pressure", "bool"),
        Field("onset_frames", "Pressure sampling frames", "int", 0, 8,
              help="Frames to wait for the contact to settle before reading pressure"),
    ]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.rows = int(self.cfg["rows"])
        self.cols = int(self.cfg["cols"])
        self._held: dict[int, int] = {}  # touch -> pad
        self._pending: dict[int, list] = {}  # touch -> [pad, frames, max pressure, contact]
        self._count: dict[int, int] = {}  # pad -> fingers holding it
        self.lit: dict[int, float] = {}  # pad -> velocity 0..1 for the overlay

    @property
    def velocity_source(self) -> str:
        src = self.cfg["velocity"]
        return "fixed" if src == "pressure" and not self.pressure_supported else src

    def pad_at(self, x: float, y: float) -> int:
        col = min(self.cols - 1, int(clamp01(x) * self.cols))
        row = min(self.rows - 1, int(clamp01(y) * self.rows))
        return row * self.cols + col

    def pad_note(self, pad: int) -> int:
        return int(self.cfg["base_note"]) + pad

    def pad_label(self, pad: int) -> str:
        n = self.pad_note(pad)
        return note_name(n) if 0 <= n <= 127 else "-"

    def on_begin(self, c: Contact) -> None:
        pad = self.pad_at(c.x, c.y)
        if self.velocity_source == "pressure" and self.cfg["onset_frames"] > 0:
            self._pending[c.id] = [pad, 0, c.pressure, c]
        else:
            self._fire(c.id, pad, c)

    def on_move(self, c: Contact) -> None:
        p = self._pending.get(c.id)
        if p is not None:
            p[1] += 1
            p[2] = max(p[2], c.pressure)
            p[3] = c
            if p[1] >= self.cfg["onset_frames"]:
                del self._pending[c.id]
                self._fire(c.id, p[0], c, p[2])
            return
        pad = self._held.get(c.id)
        if pad is not None and self.cfg["aftertouch"] and self.pressure_supported:
            self.midi.poly_pressure(self.channel, self.pad_note(pad), c.pressure * 127)

    def on_end(self, c: Contact) -> None:
        p = self._pending.pop(c.id, None)
        if p is not None:  # a very short tap: fire with what we saw
            self._fire(c.id, p[0], p[3], p[2])
        pad = self._held.pop(c.id, None)
        if pad is None:
            return
        self._count[pad] -= 1
        if self._count[pad] <= 0:
            del self._count[pad]
            self.lit.pop(pad, None)
            self.midi.note_off(self.channel, self.pad_note(pad))

    def _velocity(self, pad: int, c: Contact, pressure: float | None) -> int:
        src = self.velocity_source
        if src == "fixed":
            return int(self.cfg["fixed_velocity"])
        if src == "position":
            row_h = 1 / self.rows
            amount = clamp01((c.y - (pad // self.cols) * row_h) / row_h)
        else:
            amount = c.pressure if pressure is None else pressure
        amount = clamp01(amount) ** (self.cfg["velocity_curve"] / 100)
        return max(1, min(127, round(1 + amount * 126)))

    def _fire(self, touch: int, pad: int, c: Contact, pressure: float | None = None) -> None:
        note = self.pad_note(pad)
        if not 0 <= note <= 127:
            return
        vel = self._velocity(pad, c, pressure)
        self._held[touch] = pad
        self._count[pad] = self._count.get(pad, 0) + 1
        self.lit[pad] = vel / 127
        self.midi.note_on(self.channel, note, vel)
