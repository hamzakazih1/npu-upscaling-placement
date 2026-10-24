from __future__ import annotations

from padmidi.modes.base import Field, Mode, clamp01
from padmidi.touch import Contact


class XYMode(Mode):
    """A two-axis pad (filter cutoff vs resonance, send vs pan...). The first
    finger down owns the pad; pressure can drive a third CC."""

    key = "xy"
    title = "XY Pad"
    defaults = {
        "channel": 1,
        "cc_x": 16,
        "cc_y": 17,
        "cc_pressure": -1,
        "spring": False,
    }
    fields = [
        Field("channel", "MIDI channel", "int", 1, 16),
        Field("cc_x", "X axis CC", "int", 0, 127),
        Field("cc_y", "Y axis CC", "int", 0, 127),
        Field("cc_pressure", "Pressure CC (-1 = off)", "int", -1, 127),
        Field("spring", "Spring back to centre on release", "bool"),
    ]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.x = 0.5
        self.y = 0.5
        self.pressure = 0.0
        self.owner: int | None = None
        self._sent: dict[int, int] = {}

    def on_begin(self, c: Contact) -> None:
        if self.owner is None:
            self.owner = c.id
            self._update(c)

    def on_move(self, c: Contact) -> None:
        if c.id == self.owner:
            self._update(c)

    def on_end(self, c: Contact) -> None:
        if c.id != self.owner:
            return
        self.owner = None
        self.pressure = 0.0
        if self.cfg["cc_pressure"] >= 0:
            self._send(self.cfg["cc_pressure"], 0)
        if self.cfg["spring"]:
            self.x = self.y = 0.5
            self._send(self.cfg["cc_x"], 64)
            self._send(self.cfg["cc_y"], 64)

    def _update(self, c: Contact) -> None:
        self.x, self.y = clamp01(c.x), clamp01(c.y)
        self._send(self.cfg["cc_x"], round(self.x * 127))
        self._send(self.cfg["cc_y"], round(self.y * 127))
        if self.cfg["cc_pressure"] >= 0 and self.pressure_supported:
            self.pressure = c.pressure
            self._send(self.cfg["cc_pressure"], round(c.pressure * 127))

    def _send(self, cc: int, v: int) -> None:
        if self._sent.get(cc) != v:
            self._sent[cc] = v
            self.midi.cc(self.channel, cc, v)
