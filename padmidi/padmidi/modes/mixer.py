from __future__ import annotations

from padmidi.modes.base import Field, Mode, clamp01
from padmidi.touch import Contact

# Mackie Control: faders are 14-bit pitch bend on channels 1-8, fader touch
# sensors are notes 0x68-0x6F, bank left/right are notes 0x2E/0x2F.
MCU_TOUCH = 0x68
MCU_BANK_LEFT = 0x2E
MCU_BANK_RIGHT = 0x2F

BANK_STRIP = 0.86  # the top 14% of the pad holds the bank buttons


class MixerMode(Mode):
    """Eight channel faders for the DAW mixer, either as Mackie Control (the
    DAW maps them to its tracks and sends positions back) or as plain CCs."""

    key = "mixer"
    title = "Mixer"
    defaults = {
        "protocol": "mackie",
        "faders": 8,
        "relative": True,
        "sensitivity": 100,
        "cc": 7,
        "first_channel": 1,
    }
    fields = [
        Field("protocol", "Protocol", "choice", options=["mackie", "cc"],
              help="mackie: add PadMIDI as a Mackie Control surface in your DAW"),
        Field("faders", "Faders", "int", 1, 8),
        Field("relative", "Relative (no jumps)", "bool"),
        Field("sensitivity", "Relative sensitivity %", "int", 10, 400),
        Field("cc", "CC number (cc protocol)", "int", 0, 127),
        Field("first_channel", "First channel (cc protocol)", "int", 1, 16),
    ]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.count = int(self.cfg["faders"])
        self.values = [0.75] * self.count
        self._owner: dict[int, tuple[int, float, float]] = {}
        self._bank_touch: dict[int, int] = {}
        self.bank_lit: set[int] = set()

    @property
    def mackie(self) -> bool:
        return self.cfg["protocol"] == "mackie"

    @property
    def fader_top(self) -> float:
        return BANK_STRIP if self.mackie else 1.0

    def fader_at(self, x: float) -> int:
        return min(self.count - 1, int(clamp01(x) * self.count))

    def active_faders(self) -> set[int]:
        return {i for i, _, _ in self._owner.values()}

    def on_begin(self, c: Contact) -> None:
        if self.mackie and c.y >= BANK_STRIP:
            note = MCU_BANK_LEFT if c.x < 0.5 else MCU_BANK_RIGHT
            self._bank_touch[c.id] = note
            self.bank_lit.add(note)
            self.midi.note_on(1, note, 127)
            return
        i = self.fader_at(c.x)
        self._owner[c.id] = (i, c.y, self.values[i])
        if self.mackie:
            self.midi.note_on(1, MCU_TOUCH + i, 127)
        if not self.cfg["relative"]:
            self._set(i, clamp01(c.y / self.fader_top))

    def on_move(self, c: Contact) -> None:
        if c.id not in self._owner:
            return
        i, y0, v0 = self._owner[c.id]
        if self.cfg["relative"]:
            delta = (c.y - y0) / self.fader_top * self.cfg["sensitivity"] / 100
            self._set(i, clamp01(v0 + delta))
        else:
            self._set(i, clamp01(c.y / self.fader_top))

    def on_end(self, c: Contact) -> None:
        note = self._bank_touch.pop(c.id, None)
        if note is not None:
            self.bank_lit.discard(note)
            self.midi.send([0x90, note, 0])
            return
        owned = self._owner.pop(c.id, None)
        if owned and self.mackie:
            self.midi.send([0x90, MCU_TOUCH + owned[0], 0])  # MCU expects note-on vel 0

    def on_midi_in(self, msg: list[int]) -> None:
        if len(msg) < 3:
            return
        kind, ch = msg[0] & 0xF0, msg[0] & 0x0F
        if self.mackie and kind == 0xE0 and ch < self.count and ch not in self.active_faders():
            self.values[ch] = ((msg[2] << 7) | msg[1]) / 16383
        elif not self.mackie and kind == 0xB0 and msg[1] == self.cfg["cc"]:
            i = ch - (int(self.cfg["first_channel"]) - 1)
            if 0 <= i < self.count and i not in self.active_faders():
                self.values[i] = msg[2] / 127

    def _set(self, i: int, value: float) -> None:
        self.values[i] = value
        if self.mackie:
            self.midi.pitch_bend(i + 1, round(value * 16383))
        else:
            ch = int(self.cfg["first_channel"]) + i
            if ch <= 16:
                self.midi.cc(ch, int(self.cfg["cc"]), value * 127)
