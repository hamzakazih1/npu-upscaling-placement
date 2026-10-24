from padmidi.modes.base import Field, Mode
from padmidi.modes.keys import KeysMode
from padmidi.modes.mixer import MixerMode
from padmidi.modes.sliders import SlidersMode
from padmidi.modes.trigger import TriggerMode
from padmidi.modes.xy import XYMode

MODE_CLASSES: list[type[Mode]] = [SlidersMode, XYMode, TriggerMode, KeysMode, MixerMode]
MODES_BY_KEY = {m.key: m for m in MODE_CLASSES}

__all__ = ["Field", "Mode", "MODE_CLASSES", "MODES_BY_KEY"]
