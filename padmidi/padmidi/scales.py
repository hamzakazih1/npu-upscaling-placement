"""Scales and note naming.

Note names follow the DAW convention used by Ableton, Logic and Cubase:
middle C (MIDI 60) is C3, so MIDI 36 (kick in a drum rack) is C1.
"""

from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "major pentatonic": [0, 2, 4, 7, 9],
    "minor pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole tone": [0, 2, 4, 6, 8, 10],
    "chromatic": list(range(12)),
}


def note_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 2}"


def root_index(root: str) -> int:
    root = root.strip().upper().replace("♯", "#")
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    root = flats.get(root, root)
    return NOTE_NAMES.index(root)


def base_note(root: str, octave: int) -> int:
    """MIDI note of ``root`` in DAW octave ``octave`` (C3 = 60)."""
    return (octave + 2) * 12 + root_index(root)


def scale_note(base: int, intervals: list[int], degree: int) -> int:
    """MIDI note of scale ``degree`` (0 = root; may be negative or > len)."""
    octave, step = divmod(degree, len(intervals))
    return base + 12 * octave + intervals[step]


def chord(base: int, intervals: list[int], degree: int, kind: str) -> list[int]:
    """Diatonic chord on ``degree`` built by stacking scale thirds."""
    if kind == "triad":
        steps = (0, 2, 4)
    elif kind == "seventh":
        steps = (0, 2, 4, 6)
    elif kind == "power":
        steps = (0, 4)
    else:
        steps = (0,)
    return [scale_note(base, intervals, degree + s) for s in steps]
