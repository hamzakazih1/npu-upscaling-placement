"""macOS trackpad input via the MultitouchSupport private framework.

This is the same raw contact stream the system gesture recogniser consumes.
It works for the built-in Force Touch trackpad and Magic Trackpads, needs no
permission prompt, and reports each finger's normalised position and contact
size (which tracks how hard the finger presses).
"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import CFUNCTYPE, POINTER, Structure, c_double, c_float, c_int, c_long, c_void_p

from padmidi.touch import Contact, FrameCallback, TouchBackend, normalise_pressure

log = logging.getLogger(__name__)

_MT_PATH = "/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport"
_CF_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"


class MTPoint(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class MTReadout(Structure):
    _fields_ = [("pos", MTPoint), ("vel", MTPoint)]


class MTFinger(Structure):
    # 96 bytes; layout matches the framework's per-contact record.
    _fields_ = [
        ("frame", c_int),
        ("timestamp", c_double),
        ("identifier", c_int),
        ("state", c_int),
        ("finger_id", c_int),
        ("hand_id", c_int),
        ("normalized", MTReadout),
        ("size", c_float),
        ("zero1", c_int),
        ("angle", c_float),
        ("major_axis", c_float),
        ("minor_axis", c_float),
        ("mm", MTReadout),
        ("zero2", c_int * 2),
        ("unk2", c_float),
    ]


# Contact states: 1 start-in-range, 2 hover, 3 make-touch, 4 touching,
# 5 break-touch, 6 linger, 7 out-of-range. Only 3 and 4 are fingers down.
_TOUCHING = (3, 4)

MTContactCallback = CFUNCTYPE(c_int, c_void_p, POINTER(MTFinger), c_int, c_double, c_int)


class MacMultitouchBackend(TouchBackend):
    name = "macOS multitouch"
    pressure_supported = True

    def __init__(self, settings: dict) -> None:
        self._mt = ctypes.CDLL(_MT_PATH)
        self._cf = ctypes.CDLL(_CF_PATH)
        self._mt.MTDeviceCreateList.restype = c_void_p
        self._mt.MTRegisterContactFrameCallback.argtypes = [c_void_p, MTContactCallback]
        self._mt.MTUnregisterContactFrameCallback.argtypes = [c_void_p, MTContactCallback]
        self._mt.MTDeviceStart.argtypes = [c_void_p, c_int]
        self._mt.MTDeviceStop.argtypes = [c_void_p]
        self._cf.CFArrayGetCount.argtypes = [c_void_p]
        self._cf.CFArrayGetCount.restype = c_long
        self._cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_long]
        self._cf.CFArrayGetValueAtIndex.restype = c_void_p

        self._p_lo = float(settings.get("pressure_min", 0.15))
        self._p_hi = float(settings.get("pressure_max", 1.05))
        self._devices: list[int] = []
        self._per_device: dict[int, list[Contact]] = {}
        self._lock = threading.Lock()
        self._callback: FrameCallback | None = None
        # Keep a reference: ctypes frees the trampoline if this is collected.
        self._c_callback = MTContactCallback(self._on_frame)

        dev_list = self._mt.MTDeviceCreateList()
        count = self._cf.CFArrayGetCount(dev_list) if dev_list else 0
        self._devices = [self._cf.CFArrayGetValueAtIndex(dev_list, i) for i in range(count)]
        if not self._devices:
            raise RuntimeError("no multitouch trackpad found")

    def _on_frame(self, device, fingers, n, timestamp, frame) -> int:
        try:
            contacts = []
            for i in range(n):
                f = fingers[i]
                if f.state not in _TOUCHING:
                    continue
                contacts.append(
                    Contact(
                        # identifiers are per device; keep them unique across pads
                        id=(self._devices.index(device) if device in self._devices else 0) * 1000
                        + f.identifier,
                        x=min(1.0, max(0.0, f.normalized.pos.x)),
                        y=min(1.0, max(0.0, f.normalized.pos.y)),
                        pressure=normalise_pressure(f.size, self._p_lo, self._p_hi),
                    )
                )
            with self._lock:
                self._per_device[device] = contacts
                merged = [c for cs in self._per_device.values() for c in cs]
            if self._callback:
                self._callback(merged)
        except Exception:
            log.exception("multitouch frame failed")
        return 0

    def start(self, callback: FrameCallback) -> None:
        self._callback = callback
        for dev in self._devices:
            self._mt.MTRegisterContactFrameCallback(dev, self._c_callback)
            self._mt.MTDeviceStart(dev, 0)

    def stop(self) -> None:
        for dev in self._devices:
            try:
                self._mt.MTUnregisterContactFrameCallback(dev, self._c_callback)
                self._mt.MTDeviceStop(dev)
            except Exception:
                pass
        self._callback = None
