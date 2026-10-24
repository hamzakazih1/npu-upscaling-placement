# PyInstaller spec: `pyinstaller packaging/padmidi.spec` from the padmidi/ folder.
# Produces dist/PadMIDI/PadMIDI.exe on Windows and dist/PadMIDI.app on macOS.
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
ICONS = Path(SPECPATH) / "build"

hidden = ["rtmidi", "rtmidi._rtmidi"]
if sys.platform == "win32":
    hidden += ["pynput.keyboard._win32", "pynput.mouse._win32"]
    icon = str(ICONS / "PadMIDI.ico")
elif sys.platform == "darwin":
    hidden += ["pynput.keyboard._darwin", "pynput.mouse._darwin"]
    icon = str(ICONS / "PadMIDI.icns")
else:
    hidden += ["pynput.keyboard._xorg", "pynput.mouse._xorg"]
    icon = None

a = Analysis(
    [str(Path(SPECPATH) / "launch.py")],
    pathex=[str(ROOT)],
    hiddenimports=hidden,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PadMIDI",
    console=False,
    icon=icon,
    target_arch=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PadMIDI")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PadMIDI.app",
        icon=icon,
        bundle_identifier="app.padmidi.PadMIDI",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "LSUIElement": True,  # menu-bar app, no Dock icon
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        },
    )
