"""MIDI output/input.

Modes talk to a ``MidiOut`` which formats messages and hands raw bytes to a
sink. The real sink is python-rtmidi; tests use ``RecordingSink``.

Channels are 1-16 everywhere in this module's public API.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Protocol

log = logging.getLogger(__name__)


def clamp7(v: float) -> int:
    return max(0, min(127, int(round(v))))


class Sink(Protocol):
    def send(self, msg: list[int]) -> None: ...


class RecordingSink:
    """Collects messages in memory (tests, and the null output)."""

    def __init__(self) -> None:
        self.messages: list[list[int]] = []

    def send(self, msg: list[int]) -> None:
        self.messages.append(list(msg))

    def clear(self) -> None:
        self.messages.clear()


class MidiOut:
    def __init__(self, sink: Sink | None = None) -> None:
        self.sink: Sink = sink or RecordingSink()
        self._lock = threading.Lock()

    def send(self, msg: list[int]) -> None:
        with self._lock:
            try:
                self.sink.send(msg)
            except Exception:  # a vanished port must never kill the touch thread
                log.exception("MIDI send failed")

    def note_on(self, ch: int, note: int, vel: int) -> None:
        if 0 <= note <= 127:
            self.send([0x90 | (ch - 1), note, max(1, clamp7(vel))])

    def note_off(self, ch: int, note: int) -> None:
        if 0 <= note <= 127:
            self.send([0x80 | (ch - 1), note, 0])

    def cc(self, ch: int, number: int, value: float) -> None:
        self.send([0xB0 | (ch - 1), number & 0x7F, clamp7(value)])

    def pitch_bend(self, ch: int, value: int) -> None:
        """value is 0..16383, 8192 = centre."""
        value = max(0, min(16383, int(value)))
        self.send([0xE0 | (ch - 1), value & 0x7F, value >> 7])

    def channel_pressure(self, ch: int, value: float) -> None:
        self.send([0xD0 | (ch - 1), clamp7(value)])

    def poly_pressure(self, ch: int, note: int, value: float) -> None:
        self.send([0xA0 | (ch - 1), note & 0x7F, clamp7(value)])

    def rpn(self, ch: int, number: int, msb: int, lsb: int | None = None) -> None:
        self.cc(ch, 101, (number >> 7) & 0x7F)
        self.cc(ch, 100, number & 0x7F)
        self.cc(ch, 6, msb)
        if lsb is not None:
            self.cc(ch, 38, lsb)
        self.cc(ch, 101, 127)  # RPN null so later data entry is harmless
        self.cc(ch, 100, 127)

    def all_notes_off(self) -> None:
        for ch in range(1, 17):
            self.cc(ch, 123, 0)


# --------------------------------------------------------------------------
# python-rtmidi backed ports


def _rtmidi():
    import rtmidi  # imported lazily so the core is testable without it

    return rtmidi


class _NullSink:
    def send(self, msg: list[int]) -> None:
        pass


class RtMidiSink:
    def __init__(self, port) -> None:
        self.port = port

    def send(self, msg: list[int]) -> None:
        self.port.send_message(msg)


VIRTUAL_SUPPORTED = sys.platform != "win32"
AUTO_PREFERENCES = ("padmidi", "loopmidi", "loopback", "virtual", "iac")


def list_output_ports() -> list[str]:
    try:
        out = _rtmidi().MidiOut()
        return list(out.get_ports())
    except Exception:
        log.exception("could not list MIDI outputs")
        return []


def list_input_ports() -> list[str]:
    try:
        inp = _rtmidi().MidiIn()
        return list(inp.get_ports())
    except Exception:
        log.exception("could not list MIDI inputs")
        return []


def pick_port(ports: list[str], wanted: str, exclude: tuple[str, ...] = ()) -> int | None:
    """Index of the best port for ``wanted`` ("" means automatic)."""
    candidates = [(i, p) for i, p in enumerate(ports) if not any(e in p.lower() for e in exclude)]
    if wanted:
        for i, p in candidates:
            if p == wanted:
                return i
        for i, p in candidates:
            if wanted.lower() in p.lower():
                return i
        return None
    for pref in AUTO_PREFERENCES:
        for i, p in candidates:
            if pref in p.lower():
                return i
    return candidates[0][0] if candidates else None


class PortManager:
    """Owns the rtmidi output (and optional feedback input) for the app.

    On macOS/Linux an automatic output is a virtual port named ``virtual_name``
    that every DAW sees as a MIDI input. Windows has no virtual ports, so the
    automatic choice is a loopMIDI / Windows MIDI Services loopback port.
    """

    def __init__(self, midi: MidiOut, virtual_name: str = "PadMIDI") -> None:
        self.midi = midi
        self.virtual_name = virtual_name
        self.feedback_name = f"{virtual_name} Feedback"
        self._out = None
        self._in = None
        self.output_label = "none"
        self.input_label = "none"

    def open_output(self, wanted: str) -> str:
        self.close_output()
        try:
            rtmidi = _rtmidi()
            out = rtmidi.MidiOut(name=self.virtual_name)
            if not wanted and VIRTUAL_SUPPORTED:
                out.open_virtual_port(self.virtual_name)
                self.output_label = f"{self.virtual_name} (virtual)"
            else:
                idx = pick_port(list(out.get_ports()), wanted)
                if idx is None:
                    self.output_label = "none"
                    self.midi.sink = _NullSink()
                    return self.output_label
                out.open_port(idx)
                self.output_label = out.get_ports()[idx]
            self._out = out
            self.midi.sink = RtMidiSink(out)
        except Exception:
            log.exception("could not open MIDI output")
            self.midi.sink = _NullSink()
            self.output_label = "none"
        return self.output_label

    def open_input(self, wanted: str, callback: Callable[[list[int]], None]) -> str:
        """Feedback input for DAW fader positions (Mixer mode)."""
        self.close_input()
        if wanted == "off":
            return "off"
        try:
            rtmidi = _rtmidi()
            inp = rtmidi.MidiIn(name=self.virtual_name)
            if not wanted and VIRTUAL_SUPPORTED:
                inp.open_virtual_port(self.feedback_name)
                self.input_label = f"{self.feedback_name} (virtual)"
            else:
                ports = list(inp.get_ports())
                idx = pick_port(ports, wanted or "feedback")
                if idx is None:
                    self.input_label = "none"
                    return self.input_label
                inp.open_port(idx)
                self.input_label = ports[idx]
            inp.set_callback(lambda event, _data=None: callback(list(event[0])))
            self._in = inp
        except Exception:
            log.exception("could not open MIDI input")
            self.input_label = "none"
        return self.input_label

    def close_output(self) -> None:
        if self._out is not None:
            try:
                self.midi.all_notes_off()
                self._out.close_port()
            except Exception:
                pass
            self._out = None
            self.midi.sink = _NullSink()

    def close_input(self) -> None:
        if self._in is not None:
            try:
                self._in.cancel_callback()
                self._in.close_port()
            except Exception:
                pass
            self._in = None

    def close(self) -> None:
        self.close_output()
        self.close_input()
