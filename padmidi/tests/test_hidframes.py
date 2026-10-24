from padmidi.midi import pick_port
from padmidi.touch.hidframes import FrameAssembler, RawSlot


def slot(cid, tip=True, x=0.5, y=0.5, **kw):
    return RawSlot(cid, tip, x, y, **kw)


def test_parallel_report_is_one_frame():
    a = FrameAssembler()
    frame = a.feed(2, [slot(0, x=0.1, y=0.0), slot(1), slot(9, tip=False)])
    assert [c.id for c in frame] == [0, 1]
    assert frame[0].y == 1.0  # HID y=0 is the top edge


def test_hybrid_reports_accumulate():
    a = FrameAssembler()
    assert a.feed(5, [slot(0), slot(1)]) is None
    assert a.feed(0, [slot(2), slot(3)]) is None
    frame = a.feed(0, [slot(4), slot(5)])  # only one of these belongs to the frame
    assert [c.id for c in frame] == [0, 1, 2, 3, 4]


def test_lift_and_palm_are_dropped():
    a = FrameAssembler()
    frame = a.feed(2, [slot(0, tip=False), slot(1, confident=False)])
    assert frame == []


def test_stray_continuation_ignored():
    assert FrameAssembler().feed(0, [slot(0)]) is None


def test_no_contact_count_usage():
    frame = FrameAssembler().feed(None, [slot(3, pressure=0.25)])
    assert frame[0].id == 3 and frame[0].pressure == 0.25


def test_pick_port():
    ports = ["Microsoft GS Wavetable Synth 0", "loopMIDI Port 1", "PadMIDI 2"]
    assert pick_port(ports, "") == 2
    assert pick_port(ports, "loopMIDI Port 1") == 1
    assert pick_port(ports, "wavetable") == 0
    assert pick_port(ports, "missing") is None
    assert pick_port(["Microsoft GS Wavetable Synth 0"], "") == 0
