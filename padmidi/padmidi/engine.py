"""Turns touch frames into mode events, thread-safely.

Touch backends call ``on_frame`` from their own thread; the overlay reads
mode state from the Qt thread. Both go through ``lock``.
"""

from __future__ import annotations

import threading

from padmidi.midi import MidiOut
from padmidi.modes import MODE_CLASSES, Mode
from padmidi.touch import Contact


class Engine:
    def __init__(self, midi: MidiOut, mode_cfgs: dict, current: str = "keys", pressure_supported: bool = True) -> None:
        self.midi = midi
        self.lock = threading.RLock()
        self.active = False
        self.pressure_supported = pressure_supported
        self.modes: list[Mode] = []
        self.index = 0
        self._prev: dict[int, Contact] = {}
        self.configure(mode_cfgs, current)

    @property
    def mode(self) -> Mode:
        return self.modes[self.index]

    def configure(self, mode_cfgs: dict, current: str | None = None) -> None:
        """(Re)build all modes from configuration."""
        with self.lock:
            key = current or (self.mode.key if self.modes else "keys")
            if self.modes:
                self.mode.release_all()
            self._prev = {}
            self.modes = [cls(mode_cfgs.get(cls.key), self.midi, self.pressure_supported) for cls in MODE_CLASSES]
            self.index = next((i for i, m in enumerate(self.modes) if m.key == key), 0)
            if self.active:
                self.mode.activate()

    def set_active(self, active: bool) -> None:
        with self.lock:
            if active == self.active:
                return
            self.active = active
            if active:
                self.mode.activate()
            else:
                self._release()

    def select(self, index: int) -> None:
        with self.lock:
            index %= len(self.modes)
            if index == self.index:
                return
            self._release()
            self.index = index
            if self.active:
                self.mode.activate()

    def select_key(self, key: str) -> None:
        for i, m in enumerate(self.modes):
            if m.key == key:
                self.select(i)

    def next_mode(self) -> None:
        self.select(self.index + 1)

    def _release(self) -> None:
        self.mode.release_all()
        self._prev = {}

    def on_frame(self, contacts: list[Contact]) -> None:
        with self.lock:
            if not self.active:
                self._prev = {}
                return
            now = {c.id: c for c in contacts}
            mode = self.mode
            for cid, c in self._prev.items():
                if cid not in now:
                    mode.end(c)
            for cid, c in now.items():
                if cid in self._prev:
                    mode.move(c)
                else:
                    mode.begin(c)
            self._prev = now

    def on_midi_in(self, msg: list[int]) -> None:
        with self.lock:
            for m in self.modes:
                m.on_midi_in(msg)
