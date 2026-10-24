# PadMIDI

Turn your laptop trackpad into an expressive MIDI controller, on **Windows and macOS**. You get sliders, an XY pad, velocity-sensitive drum pads, scale-locked keys with MPE, and a mixer, all played with your fingers on the trackpad and sent to any DAW.

It runs from the tray / menu bar. Press **Ctrl+Alt+M** to start playing: the pointer freezes and a small overlay shows what is under each finger. Press it again to get your mouse back.

| Mode | What it does |
|---|---|
| **Sliders** | Up to 8 vertical CC faders across the pad, e.g. Expression (CC11), Dynamics (CC1), Reverb, Release for orchestral libraries. Absolute or relative (no jumps). |
| **XY Pad** | Two CCs from finger position, plus an optional third from pressure. Can spring back to centre on release. |
| **Trigger** | A grid of drum / clip pads (default 6×4 from C1, the kick in a drum rack). Velocity comes from how firmly you tap, where you tap in the pad, or a fixed value. Optional polyphonic aftertouch. |
| **Scale** | Keys locked to a scale (14 scales, any root), one octave per row, so nothing you play is out of key. Optional diatonic chords (triads, sevenths, power). **MPE** gives each finger its own channel: slide sideways to glide through the scale, slide up/down for timbre (CC74), press for pressure. |
| **Mixer** | 8 channel faders as a **Mackie Control** surface (with bank buttons and fader feedback from the DAW) or as plain CCs. |

Works with Logic Pro, Ableton Live, Pro Tools, Reaper, Cubase, Bitwig Studio, Studio One, FL Studio, LUNA, Digital Performer: anything that takes MIDI input.

## Download

The [PadMIDI builds](../../actions/workflows/padmidi.yml) workflow builds both apps on every push. Open the latest run and download **PadMIDI-Windows** or **PadMIDI-macOS** from Artifacts. Pushing a tag such as `padmidi-v0.1.0` also attaches both zips to a GitHub Release.

The builds are not code-signed:

- **macOS:** unzip, move `PadMIDI.app` to Applications, then right-click it and choose **Open** the first time. If macOS says it is damaged, run `xattr -cr /Applications/PadMIDI.app`. Requires macOS 12 or later on Apple Silicon (Intel Macs can run it from source).
- **Windows:** unzip anywhere and run `PadMIDI\PadMIDI.exe`. SmartScreen may ask you to confirm with *More info → Run anyway*. Requires Windows 10/11 with a **Precision Touchpad** (almost every laptop since 2017).

## Setup

### macOS

1. Start PadMIDI. It appears in the menu bar and creates a virtual MIDI port called **PadMIDI**.
2. In your DAW, enable **PadMIDI** as a MIDI input. Logic and most DAWs pick it up on their own.
3. For the global hotkeys, allow PadMIDI under *System Settings → Privacy & Security → Input Monitoring* (and *Accessibility* if asked). You can still activate it from the menu bar or the overlay's power button without them.
4. Recommended: turn off three- and four-finger swipe gestures in *System Settings → Trackpad → More Gestures* so they don't switch Spaces while you play.

### Windows

Windows has no built-in virtual MIDI ports, so PadMIDI sends to a loopback port:

1. Install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) (free) and add a port named **PadMIDI**. On Windows 11 with Windows MIDI Services you can create a loopback endpoint instead.
2. Start PadMIDI. It finds the port on its own. You can also pick any output in *Settings → MIDI output*.
3. In your DAW, enable **PadMIDI** as a MIDI input.
4. Recommended: in *Settings → Bluetooth & devices → Touchpad*, set three- and four-finger gestures to *Nothing*.

### Mixer mode (Mackie Control)

Add a **Mackie Control** control surface in your DAW. Set its input to **PadMIDI** and its output to **PadMIDI Feedback**. On macOS that is a virtual port PadMIDI creates. On Windows, create a second loopMIDI port with that name. With feedback connected, the faders follow the DAW, and the top strip of the pad banks through tracks 8 at a time.

## Using it

- **Ctrl+Alt+M** activates or pauses. **Ctrl+Alt+N** goes to the next mode. Both can be changed in Settings.
- Click a mode tab on the overlay, or choose a mode from the tray / menu-bar menu.
- *Settings…* in the tray menu has every option for every mode: CC numbers, channels, root and scale, grid size, velocity source, MPE, overlay position and size.
- Settings are saved in `~/Library/Application Support/PadMIDI/config.json` on macOS and `%APPDATA%\PadMIDI\config.json` on Windows.
- Without a supported trackpad (or with `--mouse`), you can play by clicking and dragging on the overlay.

### Pressure

On a Mac the contact size of each finger (how flat it presses) is used as pressure. It drives velocity, aftertouch and MPE pressure. If light taps come out too loud or too quiet, adjust *Pressure: lightest / firmest touch* in Settings.

On Windows, pressure is used only when the touchpad reports it (some haptic/force pads do). Otherwise velocity falls back to the fixed value. Set the velocity source to *position* to control loudness by where you strike the pad.

## How it works

| | macOS | Windows |
|---|---|---|
| Finger input | `MultitouchSupport.framework`: raw contacts with position and size from the built-in or Magic Trackpad | Raw Input HID reports from the Precision Touchpad (usage page 0x0D / 0x05), decoded with `HidP_*`, with hybrid-report assembly |
| MIDI | CoreMIDI virtual ports via python-rtmidi | WinMM ports via python-rtmidi (loopMIDI / Windows MIDI Services) |
| Pointer freeze | `CGAssociateMouseAndMouseCursorPosition` | `ClipCursor` |

```
padmidi/
  touch/      macos.py, windows.py, hidframes.py: per-OS finger input → frames
  modes/      sliders, xy, trigger, keys (scale), mixer: touches → MIDI (no Qt)
  engine.py   diffs frames into began/moved/ended, thread-safe
  midi.py     message formatting and rtmidi ports
  ui/         overlay, painters, settings dialog
  app.py      tray, hotkeys, wiring
```

## Run from source

```bash
cd padmidi
python -m pip install -r requirements.txt
python -m padmidi            # --mouse to play with the mouse, --debug for logs
python -m pip install pytest && python -m pytest
```

To build the app locally:

```bash
pip install pyinstaller pillow
python packaging/make_icons.py packaging/build
pyinstaller --noconfirm packaging/padmidi.spec   # dist/PadMIDI.app or dist/PadMIDI/PadMIDI.exe
```
