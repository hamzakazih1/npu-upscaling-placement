"""The floating overlay: a live picture of the trackpad and what it plays."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from padmidi.engine import Engine
from padmidi.touch import Contact
from padmidi.ui import painters

HEADER = 34
MARGIN = 10
TRACKPAD_ASPECT = 1.6  # width / height of a MacBook or typical PC trackpad

MOUSE_ID = 1_000_000


class Overlay(QWidget):
    mode_clicked = Signal(int)
    power_clicked = Signal()

    def __init__(self, engine: Engine, settings: dict) -> None:
        super().__init__(
            None,
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus,
        )
        self.engine = engine
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("PadMIDI")
        self.mouse_input = False  # True when no native trackpad backend exists
        self.status = ""
        self._mouse_down = False
        self.apply_settings(settings)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # -- geometry --------------------------------------------------------

    def apply_settings(self, settings: dict) -> None:
        self.settings = settings
        w = int(settings.get("width", 520))
        h = int(HEADER + (w - 2 * MARGIN) / TRACKPAD_ASPECT + MARGIN)
        self.setFixedSize(w, h)
        self.setWindowOpacity(max(0.3, min(1.0, settings.get("opacity", 92) / 100)))
        self.place()

    def place(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        pos = self.settings.get("position", "bottom")
        gap = 24
        x = area.center().x() - self.width() // 2
        if pos.endswith("left"):
            x = area.left() + gap
        elif pos.endswith("right"):
            x = area.right() - self.width() - gap
        y = area.top() + gap if pos.startswith("top") else area.bottom() - self.height() - gap
        self.move(x, y)

    def pad_rect(self) -> QRectF:
        return QRectF(MARGIN, HEADER, self.width() - 2 * MARGIN, self.height() - HEADER - MARGIN)

    def pad_centre_global(self) -> QPoint:
        return self.mapToGlobal(self.pad_rect().center().toPoint())

    def _tab_rects(self) -> list[QRectF]:
        n = len(self.engine.modes)
        left, width = MARGIN + 30, self.width() - 2 * MARGIN - 30 - 150
        return [QRectF(left + i * width / n, 6, width / n - 4, HEADER - 12) for i in range(n)]

    def _power_rect(self) -> QRectF:
        return QRectF(MARGIN, 7, 22, 22)

    # -- painting --------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(painters.LINE, 1))
        p.setBrush(painters.BG)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)

        with self.engine.lock:
            active = self.engine.active
            # power button
            pr = self._power_rect()
            p.setPen(Qt.NoPen)
            p.setBrush(painters.HOT if active else painters.CELL)
            p.drawEllipse(pr)
            p.setPen(QPen(painters.TEXT, 2))
            p.drawArc(pr.adjusted(6, 6, -6, -6), 120 * 16, 300 * 16)
            p.drawLine(QPointF(pr.center().x(), pr.top() + 4), QPointF(pr.center().x(), pr.center().y()))

            f = QFont(p.font())
            f.setPixelSize(12)
            for i, (m, tr) in enumerate(zip(self.engine.modes, self._tab_rects())):
                cur = i == self.engine.index
                p.setPen(Qt.NoPen)
                p.setBrush(painters.ACCENT if cur else painters.CELL_ALT)
                p.drawRoundedRect(tr, 6, 6)
                f.setBold(cur)
                p.setFont(f)
                p.setPen(painters.TEXT if cur else painters.MUTED)
                p.drawText(tr, Qt.AlignCenter, m.title)

            f.setBold(False)
            f.setPixelSize(10)
            p.setFont(f)
            p.setPen(painters.MUTED)
            p.drawText(QRectF(self.width() - MARGIN - 146, 4, 146, HEADER - 8),
                       Qt.AlignRight | Qt.AlignVCenter | Qt.TextWordWrap, self.status)

            r = self.pad_rect()
            painters.paint_mode(p, r, self.engine.mode)
            if not active:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, 120))
                p.drawRoundedRect(r, 8, 8)
                p.setPen(painters.TEXT)
                f.setPixelSize(13)
                p.setFont(f)
                p.drawText(r, Qt.AlignCenter, "Paused — press the power button or the toggle hotkey")

    # -- mouse: tabs always, playing only without a native trackpad -------

    def _contact(self, pos: QPointF) -> Contact:
        r = self.pad_rect()
        return Contact(MOUSE_ID, (pos.x() - r.left()) / r.width(), (r.bottom() - pos.y()) / r.height(), 0.7)

    def mousePressEvent(self, e) -> None:
        pos = e.position()
        if self._power_rect().contains(pos):
            self.power_clicked.emit()
            return
        for i, tr in enumerate(self._tab_rects()):
            if tr.contains(pos):
                self.mode_clicked.emit(i)
                return
        if self.mouse_input and self.pad_rect().contains(pos):
            self._mouse_down = True
            self.engine.on_frame([self._contact(pos)])

    def mouseMoveEvent(self, e) -> None:
        if self._mouse_down:
            self.engine.on_frame([self._contact(e.position())])

    def mouseReleaseEvent(self, e) -> None:
        if self._mouse_down:
            self._mouse_down = False
            self.engine.on_frame([])
