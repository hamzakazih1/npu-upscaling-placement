"""Common machinery for the five modes.

A mode receives touch events in trackpad coordinates (x 0..1 left to right,
y 0..1 bottom to top) and turns them into MIDI. Modes know nothing about Qt;
``padmidi.ui.painters`` draws them from their public state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from padmidi.midi import MidiOut
from padmidi.touch import Contact


@dataclass
class Field:
    """One user-editable setting, used to build the settings dialog."""

    key: str
    label: str
    kind: str  # "int", "bool", "choice", "text", "note"
    lo: int = 0
    hi: int = 127
    options: list[str] = field(default_factory=list)
    help: str = ""


def clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def fader_position(y: float, margin: float = 0.06) -> float:
    """Map a finger's y to 0..1 with a small dead zone at each edge, so the
    extremes are reachable without the finger leaving the pad."""
    return clamp01((y - margin) / (1 - 2 * margin))


class Mode:
    key = ""
    title = ""
    defaults: dict[str, Any] = {}
    fields: list[Field] = []

    def __init__(self, cfg: dict | None, midi: MidiOut, pressure_supported: bool = True) -> None:
        self.cfg = copy.deepcopy(self.defaults)
        if cfg:
            self.cfg.update(cfg)
        self.midi = midi
        self.pressure_supported = pressure_supported
        self.touches: dict[int, Contact] = {}

    @property
    def channel(self) -> int:
        return int(self.cfg.get("channel", 1))

    # Engine entry points -------------------------------------------------

    def begin(self, c: Contact) -> None:
        self.touches[c.id] = c
        self.on_begin(c)

    def move(self, c: Contact) -> None:
        self.touches[c.id] = c
        self.on_move(c)

    def end(self, c: Contact) -> None:
        self.on_end(c)
        self.touches.pop(c.id, None)

    def activate(self) -> None:
        """Called when the mode becomes current."""

    def release_all(self) -> None:
        for c in list(self.touches.values()):
            self.end(c)

    def on_midi_in(self, msg: list[int]) -> None:
        """Feedback from the DAW (only Mixer uses it)."""

    # Subclass hooks ------------------------------------------------------

    def on_begin(self, c: Contact) -> None:
        pass

    def on_move(self, c: Contact) -> None:
        pass

    def on_end(self, c: Contact) -> None:
        pass
