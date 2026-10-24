from padmidi.engine import Engine
from padmidi.midi import MidiOut, RecordingSink
from padmidi.modes.keys import KeysMode
from padmidi.modes.mixer import MCU_BANK_RIGHT, MCU_TOUCH, MixerMode
from padmidi.modes.sliders import SlidersMode, parse_sliders
from padmidi.modes.trigger import TriggerMode
from padmidi.modes.xy import XYMode
from padmidi.scales import base_note, chord, note_name, SCALES
from padmidi.touch import Contact


def out():
    sink = RecordingSink()
    return MidiOut(sink), sink


def test_note_names_use_daw_convention():
    assert note_name(60) == "C3"
    assert note_name(36) == "C1"
    assert base_note("C", 3) == 60
    assert base_note("A", 3) == 69


def test_diatonic_chords():
    major = SCALES["major"]
    assert chord(60, major, 0, "triad") == [60, 64, 67]
    assert chord(60, major, 1, "triad") == [62, 65, 69]  # D minor
    assert chord(60, major, 4, "seventh") == [67, 71, 74, 77]  # G7


def test_sliders_absolute_and_dedup():
    midi, sink = out()
    m = SlidersMode({"sliders": "Expr:11, Dyn:1"}, midi)
    m.begin(Contact(1, 0.1, 0.5))
    m.move(Contact(1, 0.1, 0.5))  # same value: nothing new
    m.move(Contact(1, 0.9, 1.0))  # stays on the slider it started on
    assert sink.messages == [[0xB0, 11, 64], [0xB0, 11, 127]]
    m.end(Contact(1, 0.9, 1.0))
    assert m.active_sliders() == set()


def test_sliders_relative():
    midi, sink = out()
    m = SlidersMode({"relative": True, "sliders": "A:20"}, midi)
    m.begin(Contact(1, 0.5, 0.2))
    assert sink.messages == []  # touching does not jump
    m.move(Contact(1, 0.5, 0.8))
    assert sink.messages[-1] == [0xB0, 20, 76]


def test_parse_sliders():
    assert parse_sliders("Expression:11,  Mod Wheel:1, bad, :7") == [
        {"label": "Expression", "cc": 11},
        {"label": "Mod Wheel", "cc": 1},
        {"label": "CC 7", "cc": 7},
    ]


def test_xy_and_spring():
    midi, sink = out()
    m = XYMode({"spring": True, "cc_pressure": 18}, midi)
    m.begin(Contact(1, 0.0, 1.0, 0.5))
    m.begin(Contact(2, 0.5, 0.5))  # second finger ignored
    assert [0xB0, 16, 0] in sink.messages and [0xB0, 17, 127] in sink.messages
    assert [0xB0, 18, 64] in sink.messages
    sink.clear()
    m.end(Contact(1, 0, 1))
    assert sink.messages == [[0xB0, 18, 0], [0xB0, 16, 64], [0xB0, 17, 64]]


def test_trigger_grid_layout():
    midi, _ = out()
    m = TriggerMode({"rows": 4, "cols": 6, "velocity": "fixed"}, midi)
    assert m.pad_label(m.pad_at(0.01, 0.01)) == "C1"
    assert m.pad_label(m.pad_at(0.99, 0.99)) == "B2"
    assert m.pad_label(m.pad_at(0.01, 0.30)) == "F#1"


def test_trigger_pressure_velocity_waits_for_onset():
    midi, sink = out()
    m = TriggerMode({"onset_frames": 2}, midi)
    m.begin(Contact(1, 0.01, 0.01, 0.1))
    assert sink.messages == []
    m.move(Contact(1, 0.01, 0.01, 0.9))
    m.move(Contact(1, 0.01, 0.01, 0.5))
    assert sink.messages == [[0x90, 36, round(1 + 0.9 * 126)]]
    m.end(Contact(1, 0.01, 0.01))
    assert sink.messages[-1] == [0x80, 36, 0]


def test_trigger_quick_tap_still_fires():
    midi, sink = out()
    m = TriggerMode({"onset_frames": 3}, midi)
    m.begin(Contact(1, 0.01, 0.01, 0.4))
    m.end(Contact(1, 0.01, 0.01))
    assert [s[0] for s in sink.messages] == [0x90, 0x80]


def test_trigger_falls_back_to_fixed_without_pressure():
    midi, sink = out()
    m = TriggerMode({"fixed_velocity": 90}, midi, pressure_supported=False)
    m.begin(Contact(1, 0.01, 0.01))
    assert sink.messages == [[0x90, 36, 90]]


def test_trigger_two_fingers_same_pad():
    midi, sink = out()
    m = TriggerMode({"velocity": "fixed"}, midi)
    m.begin(Contact(1, 0.01, 0.01))
    m.begin(Contact(2, 0.02, 0.02))
    m.end(Contact(1, 0, 0))
    assert [0x80, 36, 0] not in sink.messages
    m.end(Contact(2, 0, 0))
    assert sink.messages[-1] == [0x80, 36, 0]


def test_keys_scale_layout():
    midi, _ = out()
    m = KeysMode({"root": "D", "scale": "natural minor", "octave": 3, "rows": 2, "cols": 8}, midi)
    assert [m.key_label(0, c) for c in range(8)] == ["D3", "E3", "F3", "G3", "A3", "A#3", "C4", "D4"]
    assert m.key_label(1, 0) == "D4"
    assert m.is_root(0, 7)


def test_keys_plain_note_and_chord():
    midi, sink = out()
    m = KeysMode({"velocity": "fixed", "chords": "triad"}, midi)
    m.begin(Contact(1, 0.01, 0.01))
    assert sink.messages == [[0x90, 60, 100], [0x90, 64, 100], [0x90, 67, 100]]
    sink.clear()
    m.end(Contact(1, 0.01, 0.01))
    assert sink.messages == [[0x80, 60, 0], [0x80, 64, 0], [0x80, 67, 0]]


def test_keys_mpe_channels_and_glide():
    midi, sink = out()
    m = KeysMode({"mpe": True, "velocity": "fixed", "bend_range": 48, "cols": 8}, midi)
    m.activate()
    sink.clear()
    m.begin(Contact(1, 0.5 / 8, 0.1))  # centre of key 0 (C3)
    m.begin(Contact(2, 2.5 / 8, 0.1))  # key 2 (E3)
    ons = [msg for msg in sink.messages if msg[0] & 0xF0 == 0x90]
    assert ons == [[0x91, 60, 100], [0x92, 64, 100]]
    sink.clear()
    m.move(Contact(1, 1.5 / 8, 0.1))  # slide onto the centre of D3: +2 semitones
    bends = [msg for msg in sink.messages if msg[0] == 0xE1]
    value = bends[-1][1] | (bends[-1][2] << 7)
    assert value == round(8192 + 2 / 48 * 8191)
    m.end(Contact(1, 0, 0))
    m.begin(Contact(3, 0.01, 0.1))
    # the freed channel goes to the back of the queue
    assert [msg for msg in sink.messages if msg[0] & 0xF0 == 0x90][-1][0] == 0x93


def test_keys_pitch_interpolates_through_scale():
    midi, _ = out()
    m = KeysMode({"cols": 8}, midi)
    assert m.pitch_at(0, 2.5 / 8) == 64  # E
    assert m.pitch_at(0, 3.0 / 8) == 64.5  # halfway E -> F
    assert m.pitch_at(0, 0.0) == 60  # clamped at the edge


def test_mixer_mackie():
    midi, sink = out()
    m = MixerMode({"protocol": "mackie", "relative": False}, midi)
    m.begin(Contact(1, 0.01, 0.43))
    assert sink.messages[0] == [0x90, MCU_TOUCH, 127]
    assert sink.messages[1][0] == 0xE0
    m.end(Contact(1, 0, 0))
    assert sink.messages[-1] == [0x90, MCU_TOUCH, 0]
    sink.clear()
    m.begin(Contact(2, 0.9, 0.95))
    m.end(Contact(2, 0.9, 0.95))
    assert sink.messages == [[0x90, MCU_BANK_RIGHT, 127], [0x90, MCU_BANK_RIGHT, 0]]


def test_mixer_feedback_updates_untouched_faders():
    midi, _ = out()
    m = MixerMode({"protocol": "mackie"}, midi)
    m.on_midi_in([0xE2, 0x7F, 0x7F])
    assert m.values[2] == 1.0


def test_mixer_cc_relative():
    midi, sink = out()
    m = MixerMode({"protocol": "cc", "cc": 7, "first_channel": 1, "faders": 8}, midi)
    m.begin(Contact(1, 3.5 / 8, 0.5))
    m.move(Contact(1, 3.5 / 8, 0.6))
    assert sink.messages[-1][:2] == [0xB3, 7]
    assert sink.messages[-1][2] == round(0.85 * 127)


def test_engine_diffs_frames_and_releases_on_pause():
    midi, sink = out()
    e = Engine(midi, {"trigger": {"velocity": "fixed"}}, "trigger")
    e.on_frame([Contact(1, 0.01, 0.01)])
    assert sink.messages == []  # inactive engine ignores touches
    e.set_active(True)
    e.on_frame([Contact(1, 0.01, 0.01)])
    e.on_frame([Contact(1, 0.01, 0.01), Contact(2, 0.99, 0.99)])
    assert [m[0] for m in sink.messages] == [0x90, 0x90]
    e.set_active(False)
    assert sorted(m[1] for m in sink.messages if m[0] == 0x80) == [36, 59]


def test_engine_switching_mode_releases_notes():
    midi, sink = out()
    e = Engine(midi, {"keys": {"velocity": "fixed"}}, "keys")
    e.set_active(True)
    e.on_frame([Contact(1, 0.01, 0.01)])
    e.next_mode()
    assert sink.messages[-1] == [0x80, 60, 0]
    assert e.mode.key == "mixer"
