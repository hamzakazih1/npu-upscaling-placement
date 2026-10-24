"""Settings window, generated from each mode's ``fields``."""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from padmidi import midi
from padmidi.modes import MODE_CLASSES, Field
from padmidi.scales import note_name


class _NoteSpin(QSpinBox):
    def textFromValue(self, v: int) -> str:
        return f"{v} ({note_name(v)})"

    def valueFromText(self, text: str) -> int:
        return int(text.split()[0])


GENERAL_FIELDS = [
    ("touch", Field("backend", "Touch input", "choice", options=["auto", "mouse"],
                    help="auto: the trackpad's raw fingers; mouse: click and drag on the overlay")),
    ("touch", Field("lock_cursor", "Freeze the pointer while active", "bool")),
    ("hotkeys", Field("toggle", "Activate / pause hotkey", "text")),
    ("hotkeys", Field("next_mode", "Next mode hotkey", "text")),
    ("overlay", Field("position", "Overlay position", "choice",
                      options=["bottom", "top", "bottom-left", "bottom-right", "top-left", "top-right"])),
    ("overlay", Field("width", "Overlay width (px)", "int", 320, 1200)),
    ("overlay", Field("opacity", "Overlay opacity %", "int", 30, 100)),
    ("overlay", Field("show_when_inactive", "Keep the overlay visible when paused", "bool")),
    ("touch", Field("pressure_min", "Pressure: lightest touch (raw)", "text")),
    ("touch", Field("pressure_max", "Pressure: firmest touch (raw)", "text")),
]


def _editor(f: Field, value):
    if f.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
    elif f.kind in ("int", "note"):
        w = _NoteSpin() if f.kind == "note" else QSpinBox()
        w.setRange(f.lo, f.hi)
        w.setValue(int(value))
    elif f.kind == "choice":
        w = QComboBox()
        w.addItems(f.options)
        if str(value) not in f.options:
            w.addItem(str(value))
        w.setCurrentText(str(value))
    else:
        w = QLineEdit(str(value))
    if f.help:
        w.setToolTip(f.help)
    return w


def _read(f: Field, w, old):
    if f.kind == "bool":
        return w.isChecked()
    if f.kind in ("int", "note"):
        return w.value()
    if f.kind == "choice":
        return w.currentText()
    text = w.text().strip()
    if isinstance(old, float):
        try:
            return float(text)
        except ValueError:
            return old
    return text


class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PadMIDI Settings")
        self.cfg = copy.deepcopy(cfg)
        self._editors: list[tuple[dict, Field, QWidget]] = []
        tabs = QTabWidget()

        general = QWidget()
        form = QFormLayout(general)
        self.out_combo = QComboBox()
        self.out_combo.addItem("Automatic", "")
        for p in midi.list_output_ports():
            self.out_combo.addItem(p, p)
        self._select(self.out_combo, cfg["midi"]["output"])
        form.addRow("MIDI output", self.out_combo)
        self.in_combo = QComboBox()
        self.in_combo.addItem("Automatic", "")
        self.in_combo.addItem("Off", "off")
        for p in midi.list_input_ports():
            self.in_combo.addItem(p, p)
        self._select(self.in_combo, cfg["midi"]["input"])
        form.addRow("Mixer feedback input", self.in_combo)
        for section, f in GENERAL_FIELDS:
            self._add(form, self.cfg[section], f)
        tabs.addTab(general, "General")

        for cls in MODE_CLASSES:
            page = QWidget()
            form = QFormLayout(page)
            mcfg = self.cfg["modes"].setdefault(cls.key, copy.deepcopy(cls.defaults))
            for f in cls.fields:
                mcfg.setdefault(f.key, cls.defaults.get(f.key))
                self._add(form, mcfg, f)
            tabs.addTab(page, cls.title)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self.resize(480, 520)

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        i = combo.findData(value)
        if i < 0:
            combo.addItem(value, value)
            i = combo.count() - 1
        combo.setCurrentIndex(i)

    def _add(self, form: QFormLayout, target: dict, f: Field) -> None:
        w = _editor(f, target.get(f.key))
        self._editors.append((target, f, w))
        form.addRow(f.label, w)

    def result_config(self) -> dict:
        for target, f, w in self._editors:
            target[f.key] = _read(f, w, target.get(f.key))
        self.cfg["midi"]["output"] = self.out_combo.currentData()
        self.cfg["midi"]["input"] = self.in_combo.currentData()
        return self.cfg
