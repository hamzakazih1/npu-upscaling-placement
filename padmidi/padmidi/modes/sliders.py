from __future__ import annotations

from padmidi.modes.base import Field, Mode, clamp01, fader_position
from padmidi.touch import Contact


def parse_sliders(spec: str) -> list[dict]:
    """"Expression:11, Dynamics:1" -> [{"label": "Expression", "cc": 11}, ...]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, cc = part.rpartition(":")
        try:
            out.append({"label": label.strip() or f"CC {int(cc)}", "cc": max(0, min(127, int(cc)))})
        except ValueError:
            continue
    return out


def format_sliders(sliders: list[dict]) -> str:
    return ", ".join(f"{s['label']}:{s['cc']}" for s in sliders)


class SlidersMode(Mode):
    """Vertical CC faders, one per column of the trackpad (expression,
    dynamics, vibrato... for orchestral libraries)."""

    key = "sliders"
    title = "Sliders"
    defaults = {
        "channel": 1,
        "relative": False,
        "sensitivity": 100,
        "sliders": "Expression:11, Dynamics:1, Reverb:91, Release:72",
    }
    fields = [
        Field("channel", "MIDI channel", "int", 1, 16),
        Field("sliders", "Sliders (label:CC, ...)", "text", help="Up to 8, e.g. Expression:11, Dynamics:1"),
        Field("relative", "Relative (pick up from current value)", "bool"),
        Field("sensitivity", "Relative sensitivity %", "int", 10, 400),
    ]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        spec = self.cfg["sliders"]
        self.sliders = (parse_sliders(spec) if isinstance(spec, str) else list(spec))[:8] or [{"label": "CC 1", "cc": 1}]
        self.values = [0.0] * len(self.sliders)
        self._sent = [-1] * len(self.sliders)
        self._owner: dict[int, tuple[int, float, float]] = {}  # touch -> (slider, y0, value0)

    def slider_at(self, x: float) -> int:
        return min(len(self.sliders) - 1, int(clamp01(x) * len(self.sliders)))

    def active_sliders(self) -> set[int]:
        return {i for i, _, _ in self._owner.values()}

    def on_begin(self, c: Contact) -> None:
        i = self.slider_at(c.x)
        self._owner[c.id] = (i, c.y, self.values[i])
        if not self.cfg["relative"]:
            self._set(i, fader_position(c.y))

    def on_move(self, c: Contact) -> None:
        if c.id not in self._owner:
            return
        i, y0, v0 = self._owner[c.id]
        if self.cfg["relative"]:
            self._set(i, clamp01(v0 + (c.y - y0) * self.cfg["sensitivity"] / 100))
        else:
            self._set(i, fader_position(c.y))

    def on_end(self, c: Contact) -> None:
        self._owner.pop(c.id, None)

    def _set(self, i: int, value: float) -> None:
        self.values[i] = value
        v = round(value * 127)
        if v != self._sent[i]:
            self._sent[i] = v
            self.midi.cc(self.channel, self.sliders[i]["cc"], v)
