"""Application wiring: tray menu, hotkeys, overlay, touch and MIDI."""

from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from padmidi import APP_NAME, __version__, config
from padmidi.engine import Engine
from padmidi.midi import MidiOut, PortManager
from padmidi.platform.cursor import CursorLock
from padmidi.touch import create_backend
from padmidi.ui.overlay import Overlay
from padmidi.ui.settings import SettingsDialog

log = logging.getLogger(__name__)


def make_icon(active: bool) -> QIcon:
    pm = QPixmap(QSize(64, 64))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(47, 209, 88) if active else QColor(61, 155, 240))
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setBrush(QColor(255, 255, 255))
    for i, h in enumerate((14, 26, 38, 22)):
        p.drawRoundedRect(13 + i * 10, 50 - h, 7, h, 3, 3)
    p.end()
    return QIcon(pm)


class _Bridge(QObject):
    """Moves hotkey callbacks (pynput thread) onto the Qt thread."""

    toggle = Signal()
    next_mode = Signal()


class PadMidiApp:
    def __init__(self, app: QApplication, force_mouse: bool = False) -> None:
        self.app = app
        self.cfg = config.load()
        self.force_mouse = force_mouse

        self.midi = MidiOut()
        self.ports = PortManager(self.midi, APP_NAME)
        self.backend = self._create_backend()
        self.engine = Engine(self.midi, self.cfg["modes"], self.cfg.get("mode", "keys"),
                             pressure_supported=bool(self.backend and self.backend.pressure_supported))
        self.cursor = CursorLock()
        self.overlay = Overlay(self.engine, self.cfg["overlay"])
        self.overlay.mode_clicked.connect(self.select_mode)
        self.overlay.power_clicked.connect(self.toggle)

        self._start_touch()
        self._open_ports()

        self.tray = QSystemTrayIcon(make_icon(False))
        self.tray.setToolTip(APP_NAME)
        self._build_menu()
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

        self.bridge = _Bridge()
        self.bridge.toggle.connect(self.toggle)
        self.bridge.next_mode.connect(self.next_mode)
        self._hotkeys = None
        self._start_hotkeys()

        self._cursor_timer = QTimer()
        self._cursor_timer.timeout.connect(self.cursor.refresh)
        self._cursor_timer.start(250)

        self._refresh()
        if self.overlay.mouse_input or self.cfg["overlay"]["show_when_inactive"]:
            self.overlay.show()
        app.aboutToQuit.connect(self.shutdown)

    # -- setup -----------------------------------------------------------

    def _create_backend(self):
        return None if self.force_mouse else create_backend(self.cfg["touch"])

    def _start_touch(self) -> None:
        if self.backend is not None:
            try:
                self.backend.start(self.engine.on_frame)
                self.engine.pressure_supported = self.backend.pressure_supported
                self.engine.configure(self.cfg["modes"])
            except Exception:
                log.exception("touch backend failed to start")
                self.backend = None
        self.overlay.mouse_input = self.backend is None

    def _open_ports(self) -> None:
        self.ports.open_output(self.cfg["midi"]["output"])
        self.ports.open_input(self.cfg["midi"]["input"], self.engine.on_midi_in)

    def _start_hotkeys(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.stop()
            self._hotkeys = None
        try:
            from pynput import keyboard

            hk = self.cfg["hotkeys"]
            mapping = {}
            if hk.get("toggle"):
                mapping[hk["toggle"]] = self.bridge.toggle.emit
            if hk.get("next_mode"):
                mapping[hk["next_mode"]] = self.bridge.next_mode.emit
            if mapping:
                self._hotkeys = keyboard.GlobalHotKeys(mapping)
                self._hotkeys.start()
        except Exception:
            log.exception("global hotkeys unavailable (on macOS allow Input Monitoring)")

    def _build_menu(self) -> None:
        menu = QMenu()
        self.act_toggle = QAction("Activate")
        self.act_toggle.triggered.connect(self.toggle)
        menu.addAction(self.act_toggle)
        menu.addSeparator()
        self.mode_group = QActionGroup(menu)
        self.mode_actions = []
        for i, m in enumerate(self.engine.modes):
            a = QAction(m.title, checkable=True)
            a.triggered.connect(lambda _=False, i=i: self.select_mode(i))
            self.mode_group.addAction(a)
            menu.addAction(a)
            self.mode_actions.append(a)
        menu.addSeparator()
        self.act_overlay = QAction("Show overlay when paused", checkable=True)
        self.act_overlay.setChecked(self.cfg["overlay"]["show_when_inactive"])
        self.act_overlay.triggered.connect(self._toggle_overlay_pref)
        menu.addAction(self.act_overlay)
        settings = QAction("Settings…")
        settings.triggered.connect(self.open_settings)
        menu.addAction(settings)
        folder = QAction("Open config folder")
        folder.triggered.connect(self._open_config_folder)
        menu.addAction(folder)
        menu.addSeparator()
        self.act_info = QAction("")
        self.act_info.setEnabled(False)
        menu.addAction(self.act_info)
        quit_ = QAction(f"Quit {APP_NAME}")
        quit_.triggered.connect(self.app.quit)
        menu.addAction(quit_)
        self.menu = menu
        self._menu_actions = [settings, folder, quit_]
        self.tray.setContextMenu(menu)

    # -- actions ---------------------------------------------------------

    def toggle(self) -> None:
        self.set_active(not self.engine.active)

    def set_active(self, active: bool) -> None:
        self.engine.set_active(active)
        if active:
            self.overlay.show()
            self.overlay.raise_()
            if self.backend is not None and self.cfg["touch"]["lock_cursor"]:
                from PySide6.QtGui import QCursor

                centre = self.overlay.pad_centre_global()
                QCursor.setPos(centre)
                self.cursor.lock(centre.x(), centre.y())
        else:
            self.cursor.unlock()
            if not (self.overlay.mouse_input or self.cfg["overlay"]["show_when_inactive"]):
                self.overlay.hide()
        self._refresh()

    def select_mode(self, index: int) -> None:
        self.engine.select(index)
        self.cfg["mode"] = self.engine.mode.key
        config.save(self.cfg)
        self._refresh()

    def next_mode(self) -> None:
        self.select_mode(self.engine.index + 1)

    def open_settings(self) -> None:
        was_active = self.engine.active
        self.set_active(False)
        dlg = SettingsDialog(self.cfg)
        if dlg.exec():
            old = self.cfg
            self.cfg = dlg.result_config()
            config.save(self.cfg)
            if old["midi"] != self.cfg["midi"]:
                self._open_ports()
            if old["touch"]["backend"] != self.cfg["touch"]["backend"]:
                if self.backend:
                    self.backend.stop()
                self.backend = self._create_backend()
                self._start_touch()
            self.engine.configure(self.cfg["modes"])
            self.overlay.apply_settings(self.cfg["overlay"])
            self.act_overlay.setChecked(self.cfg["overlay"]["show_when_inactive"])
            self._start_hotkeys()
        if was_active:
            self.set_active(True)
        self._refresh()

    def _toggle_overlay_pref(self, checked: bool) -> None:
        self.cfg["overlay"]["show_when_inactive"] = checked
        config.save(self.cfg)
        if checked:
            self.overlay.show()
        elif not self.engine.active and not self.overlay.mouse_input:
            self.overlay.hide()

    def _open_config_folder(self) -> None:
        config.config_dir().mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.config_dir())))

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger and sys.platform == "win32":
            self.toggle()

    def _refresh(self) -> None:
        active = self.engine.active
        self.act_toggle.setText("Pause" if active else "Activate")
        for i, a in enumerate(self.mode_actions):
            a.setChecked(i == self.engine.index)
        source = self.backend.name if self.backend else "mouse (no trackpad)"
        self.act_info.setText(f"MIDI: {self.ports.output_label}  ·  input: {source}")
        self.overlay.status = f"MIDI → {self.ports.output_label}"
        self.tray.setIcon(make_icon(active))
        self.tray.setToolTip(f"{APP_NAME} — {self.engine.mode.title} ({'active' if active else 'paused'})")
        if self.ports.output_label == "none" and not getattr(self, "_warned", False):
            self._warned = True
            QTimer.singleShot(500, self._warn_no_port)

    def _warn_no_port(self) -> None:
        QMessageBox.warning(
            None,
            APP_NAME,
            "No MIDI output port is available.\n\n"
            "On Windows, install loopMIDI (or Windows MIDI Services) and create a port "
            "named “PadMIDI”, then pick it in Settings → MIDI output.",
        )

    def shutdown(self) -> None:
        self.cursor.unlock()
        self.engine.set_active(False)
        if self.backend:
            self.backend.stop()
        if self._hotkeys:
            self._hotkeys.stop()
        self.ports.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="padmidi", description="Trackpad MIDI controller")
    parser.add_argument("--mouse", action="store_true", help="play with the mouse on the overlay")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(make_icon(False))
    PadMidiApp(app, force_mouse=args.mouse)
    return app.exec()
