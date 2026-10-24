from __future__ import annotations

import math

from padmidi.modes.base import Field, Mode, clamp01
from padmidi.scales import NOTE_NAMES, SCALES, base_note, chord, note_name, scale_note
from padmidi.touch import Contact

MPE_MEMBER_CHANNELS = list(range(2, 17))


class KeysMode(Mode):
    """Scale-locked keys (AudioSwift calls this "Scale"). Every column is a
    note of the chosen scale, so nothing played is out of key; each row up is
    one octave higher. With MPE on, each finger gets its own channel: sliding
    sideways glides the pitch through the scale, sliding up and down sends
    CC74 and pressing harder sends channel pressure."""

    key = "keys"
    title = "Scale"
    defaults = {
        "channel": 1,
        "root": "C",
        "scale": "major",
        "octave": 3,
        "rows": 2,
        "cols": 8,
        "chords": "off",
        "velocity": "pressure",
        "fixed_velocity": 100,
        "mpe": False,
        "glide": True,
        "bend_range": 48,
        "timbre_cc": 74,
    }
    fields = [
        Field("root", "Root", "choice", options=NOTE_NAMES),
        Field("scale", "Scale", "choice", options=list(SCALES)),
        Field("octave", "Octave (C3 = middle C)", "int", -1, 7),
        Field("rows", "Rows (octaves)", "int", 1, 4),
        Field("cols", "Keys per row", "int", 3, 16),
        Field("chords", "Chords", "choice", options=["off", "triad", "seventh", "power"]),
        Field("velocity", "Velocity from", "choice", options=["pressure", "position", "fixed"]),
        Field("fixed_velocity", "Fixed velocity", "int", 1, 127),
        Field("channel", "MIDI channel (non-MPE)", "int", 1, 16),
        Field("mpe", "MPE (per-finger channels 2-16)", "bool"),
        Field("glide", "Glide pitch when sliding sideways (MPE)", "bool"),
        Field("bend_range", "MPE pitch bend range (semitones)", "int", 1, 96),
        Field("timbre_cc", "Vertical slide CC (74 = MPE timbre)", "int", 0, 127),
    ]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.intervals = SCALES.get(self.cfg["scale"], SCALES["major"])
        self.base = base_note(self.cfg["root"], int(self.cfg["octave"]))
        self.rows = int(self.cfg["rows"])
        self.cols = int(self.cfg["cols"])
        # touch -> {"degree", "row", "notes", "channel", "x0"}
        self.voices: dict[int, dict] = {}
        self._free_channels = list(MPE_MEMBER_CHANNELS)
        self._sent: dict[tuple[int, str], int] = {}

    # -- layout ----------------------------------------------------------

    def degree(self, row: int, col: int) -> int:
        return row * len(self.intervals) + col

    def key_note(self, row: int, col: int) -> int:
        return scale_note(self.base, self.intervals, self.degree(row, col))

    def key_label(self, row: int, col: int) -> str:
        n = self.key_note(row, col)
        return note_name(n) if 0 <= n <= 127 else "-"

    def is_root(self, row: int, col: int) -> bool:
        return self.degree(row, col) % len(self.intervals) == 0

    def cell(self, x: float, y: float) -> tuple[int, int]:
        col = min(self.cols - 1, int(clamp01(x) * self.cols))
        row = min(self.rows - 1, int(clamp01(y) * self.rows))
        return row, col

    def pitch_at(self, row: int, x: float) -> float:
        """Continuous pitch for a finger at ``x``: exact on key centres and
        interpolated through the scale in between."""
        c = clamp01(x) * self.cols - 0.5
        c = max(0.0, min(self.cols - 1.0, c))
        lo = math.floor(c)
        hi = min(self.cols - 1, lo + 1)
        a, b = self.key_note(row, lo), self.key_note(row, hi)
        return a + (b - a) * (c - lo)

    # -- events ----------------------------------------------------------

    def activate(self) -> None:
        if not self.cfg["mpe"]:
            return
        # MPE Configuration Message: lower zone with 15 member channels,
        # then the per-note pitch bend range on each member channel.
        self.midi.rpn(1, 6, 15)
        for ch in MPE_MEMBER_CHANNELS:
            self.midi.rpn(ch, 0, int(self.cfg["bend_range"]), 0)

    def on_begin(self, c: Contact) -> None:
        row, col = self.cell(c.x, c.y)
        notes = [n for n in chord(self.base, self.intervals, self.degree(row, col), self.cfg["chords"]) if 0 <= n <= 127]
        if not notes:
            return
        mpe = self.cfg["mpe"]
        ch = self._take_channel() if mpe else self.channel
        vel = self._velocity(c, row)
        voice = {"row": row, "col": col, "notes": notes, "channel": ch, "x0": c.x}
        self.voices[c.id] = voice
        if mpe:
            self.midi.pitch_bend(ch, 8192)
            self._send_cc(c.id, ch, self._timbre(c, row))
        for n in notes:
            self.midi.note_on(ch, n, vel)

    def on_move(self, c: Contact) -> None:
        v = self.voices.get(c.id)
        if v is None:
            return
        ch = v["channel"]
        if self.cfg["mpe"]:
            if self.cfg["glide"]:
                bend = self.pitch_at(v["row"], c.x) - self.key_note(v["row"], v["col"])
                # glide only once the finger has really moved, so a wobble on
                # the key centre never detunes the note
                if abs(c.x - v["x0"]) < 0.25 / self.cols:
                    bend = 0.0
                rng = float(self.cfg["bend_range"])
                value = round(8192 + max(-1.0, min(1.0, bend / rng)) * 8191)
                if self._sent.get((c.id, "bend")) != value:
                    self._sent[(c.id, "bend")] = value
                    self.midi.pitch_bend(ch, value)
            self._send_cc(c.id, ch, self._timbre(c, v["row"]))
            if self.pressure_supported:
                p = round(c.pressure * 127)
                if self._sent.get((c.id, "pressure")) != p:
                    self._sent[(c.id, "pressure")] = p
                    self.midi.channel_pressure(ch, p)
        elif self.pressure_supported:
            p = round(c.pressure * 127)
            if self._sent.get((c.id, "pressure")) != p:
                self._sent[(c.id, "pressure")] = p
                for n in v["notes"]:
                    self.midi.poly_pressure(ch, n, p)

    def on_end(self, c: Contact) -> None:
        v = self.voices.pop(c.id, None)
        for k in [k for k in self._sent if k[0] == c.id]:
            del self._sent[k]
        if v is None:
            return
        for n in v["notes"]:
            self.midi.note_off(v["channel"], n)
        if self.cfg["mpe"]:
            self._free_channels.append(v["channel"])

    # -- helpers ---------------------------------------------------------

    def _take_channel(self) -> int:
        # Least recently released first, so release tails are not cut off.
        if self._free_channels:
            return self._free_channels.pop(0)
        # More fingers than channels: steal the oldest voice's channel.
        oldest = next(iter(self.voices))
        ch = self.voices[oldest]["channel"]
        for n in self.voices.pop(oldest)["notes"]:
            self.midi.note_off(ch, n)
        return ch

    def _row_fraction(self, c: Contact, row: int) -> float:
        h = 1 / self.rows
        return clamp01((c.y - row * h) / h)

    def _timbre(self, c: Contact, row: int) -> int:
        return round(self._row_fraction(c, row) * 127)

    def _send_cc(self, touch: int, ch: int, value: int) -> None:
        if self._sent.get((touch, "cc")) != value:
            self._sent[(touch, "cc")] = value
            self.midi.cc(ch, int(self.cfg["timbre_cc"]), value)

    def _velocity(self, c: Contact, row: int) -> int:
        src = self.cfg["velocity"]
        if src == "pressure" and not self.pressure_supported:
            src = "fixed"
        if src == "fixed":
            return int(self.cfg["fixed_velocity"])
        amount = c.pressure if src == "pressure" else self._row_fraction(c, row)
        return max(1, min(127, round(1 + clamp01(amount) * 126)))
