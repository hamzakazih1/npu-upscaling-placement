"""Drawing each mode on the overlay.

``rect`` is the on-screen image of the trackpad; trackpad y grows upwards,
screen y grows downwards, hence ``_y``.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from padmidi.modes import Mode
from padmidi.modes.keys import KeysMode
from padmidi.modes.mixer import BANK_STRIP, MCU_BANK_LEFT, MCU_BANK_RIGHT, MixerMode
from padmidi.modes.sliders import SlidersMode
from padmidi.modes.trigger import TriggerMode
from padmidi.modes.xy import XYMode

BG = QColor(30, 31, 36)
CELL = QColor(46, 48, 56)
CELL_ALT = QColor(40, 42, 49)
LINE = QColor(70, 73, 84)
TEXT = QColor(225, 228, 235)
MUTED = QColor(140, 145, 158)
ACCENT = QColor(61, 155, 240)
ACCENT_SOFT = QColor(61, 155, 240, 110)
HOT = QColor(47, 209, 88)
ROOT = QColor(58, 64, 92)


def _x(r: QRectF, x: float) -> float:
    return r.left() + x * r.width()


def _y(r: QRectF, y: float) -> float:
    return r.bottom() - y * r.height()


def _cell(r: QRectF, x0: float, y0: float, x1: float, y1: float, inset: float = 2.0) -> QRectF:
    """Screen rect for trackpad box (x0,y0)-(x1,y1), y up."""
    return QRectF(_x(r, x0) + inset, _y(r, y1) + inset, (x1 - x0) * r.width() - 2 * inset,
                  (y1 - y0) * r.height() - 2 * inset)


def _font(p: QPainter, px: int, bold: bool = False) -> None:
    f = QFont(p.font())
    f.setPixelSize(px)
    f.setBold(bold)
    p.setFont(f)


def paint_mode(p: QPainter, r: QRectF, mode: Mode) -> None:
    p.setRenderHint(QPainter.Antialiasing)
    painter = {
        SlidersMode: _sliders,
        XYMode: _xy,
        TriggerMode: _trigger,
        KeysMode: _keys,
        MixerMode: _mixer,
    }[type(mode)]
    painter(p, r, mode)
    _touches(p, r, mode)


def _touches(p: QPainter, r: QRectF, mode: Mode) -> None:
    for c in mode.touches.values():
        radius = 7 + 9 * c.pressure
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        p.setBrush(QColor(255, 255, 255, 60))
        p.drawEllipse(QPointF(_x(r, c.x), _y(r, c.y)), radius, radius)


def _fader(p: QPainter, box: QRectF, value: float, active: bool, label: str, value_text: str) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(CELL)
    p.drawRoundedRect(box, 6, 6)
    fill_h = box.height() * value
    fill = QRectF(box.left(), box.bottom() - fill_h, box.width(), fill_h)
    p.setBrush(HOT if active else ACCENT_SOFT)
    p.drawRoundedRect(fill, 6, 6)
    p.setPen(TEXT)
    _font(p, 12, True)
    p.drawText(box.adjusted(4, 6, -4, 0), Qt.AlignHCenter | Qt.AlignTop, label)
    _font(p, 18)
    p.drawText(box.adjusted(0, 26, 0, 0), Qt.AlignHCenter | Qt.AlignTop, value_text)


def _sliders(p: QPainter, r: QRectF, m: SlidersMode) -> None:
    n = len(m.sliders)
    active = m.active_sliders()
    for i, s in enumerate(m.sliders):
        box = _cell(r, i / n, 0, (i + 1) / n, 1, 4)
        _fader(p, box, m.values[i], i in active, s["label"], str(round(m.values[i] * 127)))
        p.setPen(QColor(255, 255, 255, 170))
        _font(p, 10)
        p.drawText(box.adjusted(0, 0, 0, -6), Qt.AlignHCenter | Qt.AlignBottom, f"CC {s['cc']}")


def _xy(p: QPainter, r: QRectF, m: XYMode) -> None:
    box = r.adjusted(4, 4, -4, -4)
    p.setPen(Qt.NoPen)
    p.setBrush(CELL)
    p.drawRoundedRect(box, 8, 8)
    p.setPen(QPen(LINE, 1, Qt.DashLine))
    p.drawLine(QPointF(box.center().x(), box.top()), QPointF(box.center().x(), box.bottom()))
    p.drawLine(QPointF(box.left(), box.center().y()), QPointF(box.right(), box.center().y()))
    cx, cy = _x(box, m.x), _y(box, m.y)
    p.setPen(QPen(ACCENT if m.owner is None else HOT, 1))
    p.drawLine(QPointF(cx, box.top()), QPointF(cx, box.bottom()))
    p.drawLine(QPointF(box.left(), cy), QPointF(box.right(), cy))
    p.setBrush(ACCENT if m.owner is None else HOT)
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(cx, cy), 9, 9)
    p.setPen(MUTED)
    _font(p, 11)
    p.drawText(box.adjusted(8, 0, 0, -6), Qt.AlignLeft | Qt.AlignBottom,
               f"X  CC {m.cfg['cc_x']}  {round(m.x * 127)}")
    p.drawText(box.adjusted(8, 6, 0, 0), Qt.AlignLeft | Qt.AlignTop,
               f"Y  CC {m.cfg['cc_y']}  {round(m.y * 127)}")
    if m.cfg["cc_pressure"] >= 0:
        p.drawText(box.adjusted(0, 6, -8, 0), Qt.AlignRight | Qt.AlignTop,
                   f"Pressure  CC {m.cfg['cc_pressure']}  {round(m.pressure * 127)}")


def _trigger(p: QPainter, r: QRectF, m: TriggerMode) -> None:
    for row in range(m.rows):
        for col in range(m.cols):
            pad = row * m.cols + col
            box = _cell(r, col / m.cols, row / m.rows, (col + 1) / m.cols, (row + 1) / m.rows, 3)
            p.setPen(Qt.NoPen)
            if pad in m.lit:
                c = QColor(HOT)
                c.setAlpha(int(90 + 165 * m.lit[pad]))
                p.setBrush(c)
            else:
                p.setBrush(CELL if (row + col) % 2 == 0 else CELL_ALT)
            p.drawRoundedRect(box, 6, 6)
            p.setPen(TEXT)
            _font(p, 12)
            p.drawText(box, Qt.AlignCenter, m.pad_label(pad))


def _keys(p: QPainter, r: QRectF, m: KeysMode) -> None:
    held = {(v["row"], v["col"]) for v in m.voices.values()}
    for row in range(m.rows):
        for col in range(m.cols):
            box = _cell(r, col / m.cols, row / m.rows, (col + 1) / m.cols, (row + 1) / m.rows, 2)
            p.setPen(Qt.NoPen)
            p.setBrush(HOT if (row, col) in held else ROOT if m.is_root(row, col) else CELL)
            p.drawRoundedRect(box, 5, 5)
            p.setPen(TEXT)
            _font(p, 12, m.is_root(row, col))
            p.drawText(box.adjusted(0, 0, 0, -6), Qt.AlignHCenter | Qt.AlignBottom, m.key_label(row, col))
    p.setPen(MUTED)
    _font(p, 10)
    tag = f"{m.cfg['root']} {m.cfg['scale']}"
    if m.cfg["chords"] != "off":
        tag += f" · {m.cfg['chords']} chords"
    if m.cfg["mpe"]:
        tag += " · MPE"
    p.drawText(r.adjusted(6, 4, -6, 0), Qt.AlignRight | Qt.AlignTop, tag)


def _mixer(p: QPainter, r: QRectF, m: MixerMode) -> None:
    top = m.fader_top
    active = m.active_faders()
    for i in range(m.count):
        box = _cell(r, i / m.count, 0, (i + 1) / m.count, top, 3)
        v = m.values[i]
        text = f"{round(v * 100)}%" if m.mackie else str(round(v * 127))
        _fader(p, box, v, i in active, str(i + 1), text)
    if m.mackie:
        for note, x0, x1, label in ((MCU_BANK_LEFT, 0, 0.5, "◀  Bank"), (MCU_BANK_RIGHT, 0.5, 1, "Bank  ▶")):
            box = _cell(r, x0, BANK_STRIP, x1, 1, 3)
            p.setPen(Qt.NoPen)
            p.setBrush(HOT if note in m.bank_lit else CELL_ALT)
            p.drawRoundedRect(box, 5, 5)
            p.setPen(TEXT)
            _font(p, 11)
            p.drawText(box, Qt.AlignCenter, label)
