"""Windows Precision Touchpad input via Raw Input.

Registers for HID digitizer / touch pad reports (usage page 0x0D, usage 0x05)
with RIDEV_INPUTSINK so contacts arrive while the DAW has focus, and decodes
each report with the device's HID preparsed data. Every finger is its own
link collection carrying X, Y, contact ID and a tip switch; tip pressure or
contact width/height are used for pressure when the hardware reports them.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import threading
from ctypes import wintypes as wt

from padmidi.touch import FrameCallback, TouchBackend, normalise_pressure
from padmidi.touch.hidframes import FrameAssembler, RawSlot, normalise

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
hid = ctypes.WinDLL("hid")

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

WM_INPUT = 0x00FF
WM_QUIT = 0x0012
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEHID = 2
RIDI_PREPARSEDDATA = 0x20000005
HIDP_STATUS_SUCCESS = 0x00110000
HIDP_INPUT = 0

PAGE_GENERIC = 0x01
PAGE_DIGITIZER = 0x0D
USAGE_X = 0x30
USAGE_Y = 0x31
USAGE_TOUCHPAD = 0x05
USAGE_TIP_SWITCH = 0x42
USAGE_CONFIDENCE = 0x47
USAGE_WIDTH = 0x48
USAGE_HEIGHT = 0x49
USAGE_CONTACT_ID = 0x51
USAGE_CONTACT_COUNT = 0x54
USAGE_TIP_PRESSURE = 0x30


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wt.USHORT),
        ("usUsage", wt.USHORT),
        ("dwFlags", wt.DWORD),
        ("hwndTarget", wt.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wt.DWORD),
        ("dwSize", wt.DWORD),
        ("hDevice", wt.HANDLE),
        ("wParam", wt.WPARAM),
    ]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wt.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", wt.HICON),
        ("hCursor", wt.HANDLE),
        ("hbrBackground", wt.HBRUSH),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
    ]


class HIDP_CAPS(ctypes.Structure):
    _fields_ = [
        ("Usage", wt.USHORT),
        ("UsagePage", wt.USHORT),
        ("InputReportByteLength", wt.USHORT),
        ("OutputReportByteLength", wt.USHORT),
        ("FeatureReportByteLength", wt.USHORT),
        ("Reserved", wt.USHORT * 17),
        ("NumberLinkCollectionNodes", wt.USHORT),
        ("NumberInputButtonCaps", wt.USHORT),
        ("NumberInputValueCaps", wt.USHORT),
        ("NumberInputDataIndices", wt.USHORT),
        ("NumberOutputButtonCaps", wt.USHORT),
        ("NumberOutputValueCaps", wt.USHORT),
        ("NumberOutputDataIndices", wt.USHORT),
        ("NumberFeatureButtonCaps", wt.USHORT),
        ("NumberFeatureValueCaps", wt.USHORT),
        ("NumberFeatureDataIndices", wt.USHORT),
    ]


class _Range(ctypes.Structure):
    _fields_ = [
        ("UsageMin", wt.USHORT),
        ("UsageMax", wt.USHORT),
        ("StringMin", wt.USHORT),
        ("StringMax", wt.USHORT),
        ("DesignatorMin", wt.USHORT),
        ("DesignatorMax", wt.USHORT),
        ("DataIndexMin", wt.USHORT),
        ("DataIndexMax", wt.USHORT),
    ]


class _NotRange(ctypes.Structure):
    _fields_ = [
        ("Usage", wt.USHORT),
        ("Reserved1", wt.USHORT),
        ("StringIndex", wt.USHORT),
        ("Reserved2", wt.USHORT),
        ("DesignatorIndex", wt.USHORT),
        ("Reserved3", wt.USHORT),
        ("DataIndex", wt.USHORT),
        ("Reserved4", wt.USHORT),
    ]


class _U(ctypes.Union):
    _fields_ = [("Range", _Range), ("NotRange", _NotRange)]


class HIDP_VALUE_CAPS(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("UsagePage", wt.USHORT),
        ("ReportID", ctypes.c_ubyte),
        ("IsAlias", ctypes.c_ubyte),
        ("BitField", wt.USHORT),
        ("LinkCollection", wt.USHORT),
        ("LinkUsage", wt.USHORT),
        ("LinkUsagePage", wt.USHORT),
        ("IsRange", ctypes.c_ubyte),
        ("IsStringRange", ctypes.c_ubyte),
        ("IsDesignatorRange", ctypes.c_ubyte),
        ("IsAbsolute", ctypes.c_ubyte),
        ("HasNull", ctypes.c_ubyte),
        ("Reserved", ctypes.c_ubyte),
        ("BitSize", wt.USHORT),
        ("ReportCount", wt.USHORT),
        ("Reserved2", wt.USHORT * 5),
        ("UnitsExp", wt.ULONG),
        ("Units", wt.ULONG),
        ("LogicalMin", wt.LONG),
        ("LogicalMax", wt.LONG),
        ("PhysicalMin", wt.LONG),
        ("PhysicalMax", wt.LONG),
        ("u", _U),
    ]


class HIDP_BUTTON_CAPS(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("UsagePage", wt.USHORT),
        ("ReportID", ctypes.c_ubyte),
        ("IsAlias", ctypes.c_ubyte),
        ("BitField", wt.USHORT),
        ("LinkCollection", wt.USHORT),
        ("LinkUsage", wt.USHORT),
        ("LinkUsagePage", wt.USHORT),
        ("IsRange", ctypes.c_ubyte),
        ("IsStringRange", ctypes.c_ubyte),
        ("IsDesignatorRange", ctypes.c_ubyte),
        ("IsAbsolute", ctypes.c_ubyte),
        ("ReportCount", wt.USHORT),
        ("Reserved2", wt.USHORT),
        ("Reserved", wt.ULONG * 9),
        ("u", _U),
    ]


assert ctypes.sizeof(HIDP_VALUE_CAPS) == 72
assert ctypes.sizeof(HIDP_BUTTON_CAPS) == 72
assert ctypes.sizeof(HIDP_CAPS) == 64

user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wt.ATOM
user32.CreateWindowExW.argtypes = [
    wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID,
]
user32.CreateWindowExW.restype = wt.HWND
user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wt.UINT, wt.UINT]
user32.RegisterRawInputDevices.restype = wt.BOOL
user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, wt.LPVOID, ctypes.POINTER(wt.UINT), wt.UINT]
user32.GetRawInputData.restype = wt.UINT
user32.GetRawInputDeviceInfoW.argtypes = [wt.HANDLE, wt.UINT, wt.LPVOID, ctypes.POINTER(wt.UINT)]
user32.GetRawInputDeviceInfoW.restype = wt.UINT
user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
user32.GetMessageW.restype = wt.BOOL
user32.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DestroyWindow.argtypes = [wt.HWND]
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HMODULE

hid.HidP_GetCaps.argtypes = [ctypes.c_void_p, ctypes.POINTER(HIDP_CAPS)]
hid.HidP_GetCaps.restype = wt.LONG
hid.HidP_GetValueCaps.argtypes = [ctypes.c_int, ctypes.POINTER(HIDP_VALUE_CAPS), ctypes.POINTER(wt.USHORT), ctypes.c_void_p]
hid.HidP_GetValueCaps.restype = wt.LONG
hid.HidP_GetButtonCaps.argtypes = [ctypes.c_int, ctypes.POINTER(HIDP_BUTTON_CAPS), ctypes.POINTER(wt.USHORT), ctypes.c_void_p]
hid.HidP_GetButtonCaps.restype = wt.LONG
hid.HidP_GetUsageValue.argtypes = [
    ctypes.c_int, wt.USHORT, wt.USHORT, wt.USHORT, ctypes.POINTER(wt.ULONG), ctypes.c_void_p,
    ctypes.c_char_p, wt.ULONG,
]
hid.HidP_GetUsageValue.restype = wt.LONG
hid.HidP_GetUsages.argtypes = [
    ctypes.c_int, wt.USHORT, wt.USHORT, ctypes.POINTER(wt.USHORT), ctypes.POINTER(wt.ULONG),
    ctypes.c_void_p, ctypes.c_char_p, wt.ULONG,
]
hid.HidP_GetUsages.restype = wt.LONG


def _usage(cap) -> int:
    return cap.Range.UsageMin if cap.IsRange else cap.NotRange.Usage


def _logical_range(cap) -> tuple[int, int]:
    lo, hi = cap.LogicalMin, cap.LogicalMax
    if hi <= lo:  # unsigned maximum read back as a negative LONG
        lo, hi = 0, (1 << cap.BitSize) - 1
    return lo, hi


class _Device:
    """HID layout of one touchpad: which link collection holds which finger."""

    def __init__(self, handle) -> None:
        size = wt.UINT(0)
        user32.GetRawInputDeviceInfoW(handle, RIDI_PREPARSEDDATA, None, ctypes.byref(size))
        self._preparsed_buf = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputDeviceInfoW(handle, RIDI_PREPARSEDDATA, self._preparsed_buf, ctypes.byref(size)) in (0, 0xFFFFFFFF):
            raise OSError("no preparsed data")
        self.pp = ctypes.cast(self._preparsed_buf, ctypes.c_void_p)

        caps = HIDP_CAPS()
        if hid.HidP_GetCaps(self.pp, ctypes.byref(caps)) != HIDP_STATUS_SUCCESS:
            raise OSError("HidP_GetCaps failed")

        n = wt.USHORT(caps.NumberInputValueCaps)
        values = (HIDP_VALUE_CAPS * max(1, n.value))()
        hid.HidP_GetValueCaps(HIDP_INPUT, values, ctypes.byref(n), self.pp)
        nb = wt.USHORT(caps.NumberInputButtonCaps)
        buttons = (HIDP_BUTTON_CAPS * max(1, nb.value))()
        hid.HidP_GetButtonCaps(HIDP_INPUT, buttons, ctypes.byref(nb), self.pp)

        self.contact_count_lc: int | None = None
        fingers: dict[int, dict] = {}
        for cap in values[: n.value]:
            u, page, lc = _usage(cap), cap.UsagePage, cap.LinkCollection
            slot = fingers.setdefault(lc, {})
            if page == PAGE_GENERIC and u in (USAGE_X, USAGE_Y):
                slot["x" if u == USAGE_X else "y"] = _logical_range(cap)
            elif page == PAGE_DIGITIZER and u == USAGE_CONTACT_ID:
                slot["id"] = True
            elif page == PAGE_DIGITIZER and u == USAGE_TIP_PRESSURE:
                slot["pressure"] = _logical_range(cap)
            elif page == PAGE_DIGITIZER and u in (USAGE_WIDTH, USAGE_HEIGHT):
                slot["w" if u == USAGE_WIDTH else "h"] = _logical_range(cap)
            elif page == PAGE_DIGITIZER and u == USAGE_CONTACT_COUNT:
                self.contact_count_lc = lc
        for cap in buttons[: nb.value]:
            if cap.UsagePage != PAGE_DIGITIZER:
                continue
            lo = cap.Range.UsageMin if cap.IsRange else cap.NotRange.Usage
            hi = cap.Range.UsageMax if cap.IsRange else lo
            if lo <= USAGE_CONFIDENCE <= hi:
                fingers.setdefault(cap.LinkCollection, {})["confidence"] = True
        # A finger collection is any link collection with both X and Y.
        self.fingers = sorted(
            ((lc, s) for lc, s in fingers.items() if "x" in s and "y" in s), key=lambda kv: kv[0]
        )
        if not self.fingers:
            raise OSError("not a multi-finger touchpad")
        self.has_pressure = any("pressure" in s or "w" in s for _, s in self.fingers)
        self.assembler = FrameAssembler()

    def _value(self, page, lc, usage, report: bytes) -> int | None:
        v = wt.ULONG(0)
        status = hid.HidP_GetUsageValue(HIDP_INPUT, page, lc, usage, ctypes.byref(v), self.pp, report, len(report))
        return v.value if status == HIDP_STATUS_SUCCESS else None

    def _buttons(self, lc, report: bytes) -> set[int] | None:
        length = wt.ULONG(16)
        usages = (wt.USHORT * 16)()
        status = hid.HidP_GetUsages(HIDP_INPUT, PAGE_DIGITIZER, lc, usages, ctypes.byref(length), self.pp, report, len(report))
        if status != HIDP_STATUS_SUCCESS:
            return None
        return set(usages[: length.value])

    def parse(self, report: bytes) -> tuple[int | None, list[RawSlot]] | None:
        count = None
        if self.contact_count_lc is not None:
            count = self._value(PAGE_DIGITIZER, self.contact_count_lc, USAGE_CONTACT_COUNT, report)
            if count is None:
                return None  # a different report ID (e.g. the mouse-mode report)
        slots = []
        for lc, s in self.fingers:
            pressed = self._buttons(lc, report)
            x = self._value(PAGE_GENERIC, lc, USAGE_X, report)
            y = self._value(PAGE_GENERIC, lc, USAGE_Y, report)
            if pressed is None or x is None or y is None:
                continue
            cid = self._value(PAGE_DIGITIZER, lc, USAGE_CONTACT_ID, report) if "id" in s else lc
            pressure = None
            if "pressure" in s:
                raw = self._value(PAGE_DIGITIZER, lc, USAGE_TIP_PRESSURE, report)
                if raw is not None:
                    pressure = normalise(raw, *s["pressure"])
            elif "w" in s:
                w = self._value(PAGE_DIGITIZER, lc, USAGE_WIDTH, report)
                h = self._value(PAGE_DIGITIZER, lc, USAGE_HEIGHT, report) if "h" in s else w
                if w is not None and h is not None:
                    pressure = (normalise(w, *s["w"]) + normalise(h, *s.get("h", s["w"]))) / 2
            slots.append(
                RawSlot(
                    contact_id=cid if cid is not None else lc,
                    tip=USAGE_TIP_SWITCH in pressed,
                    x=normalise(x, *s["x"]),
                    y=normalise(y, *s["y"]),
                    pressure=pressure,
                    confident=(USAGE_CONFIDENCE in pressed) if "confidence" in s else True,
                )
            )
        return count, slots


class WindowsPrecisionTouchpadBackend(TouchBackend):
    name = "Windows Precision Touchpad"

    def __init__(self, settings: dict) -> None:
        self._p_lo = float(settings.get("pressure_min", 0.0))
        self._p_hi = float(settings.get("pressure_max", 1.0))
        self._callback: FrameCallback | None = None
        self._devices: dict[int, _Device | None] = {}
        self._frames: dict[int, list] = {}
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: str | None = None
        self._wndproc = WNDPROC(self._proc)  # keep alive
        self.pressure_supported = False

    # -- message thread -------------------------------------------------

    def start(self, callback: FrameCallback) -> None:
        self._callback = callback
        self._thread = threading.Thread(target=self._run, name="padmidi-rawinput", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self._error:
            raise OSError(self._error)

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._callback = None

    def _run(self) -> None:
        try:
            self._thread_id = kernel32.GetCurrentThreadId()
            hinst = kernel32.GetModuleHandleW(None)
            wc = WNDCLASSW()
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = hinst
            wc.lpszClassName = "PadMIDIRawInput"
            user32.RegisterClassW(ctypes.byref(wc))
            # A hidden top-level window: RIDEV_INPUTSINK needs a real target.
            hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "PadMIDI touch", 0, 0, 0, 0, 0, None, None, hinst, None)
            if not hwnd:
                raise OSError(f"CreateWindowEx failed ({ctypes.get_last_error()})")
            dev = RAWINPUTDEVICE(PAGE_DIGITIZER, USAGE_TOUCHPAD, RIDEV_INPUTSINK, hwnd)
            if not user32.RegisterRawInputDevices(ctypes.byref(dev), 1, ctypes.sizeof(dev)):
                raise OSError(f"RegisterRawInputDevices failed ({ctypes.get_last_error()})")
        except Exception as e:
            self._error = str(e)
            self._ready.set()
            return
        self._ready.set()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.DestroyWindow(hwnd)

    def _proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            try:
                self._on_input(lparam)
            except Exception:
                log.exception("raw input decode failed")
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _on_input(self, lparam) -> None:
        hdr_size = ctypes.sizeof(RAWINPUTHEADER)
        size = wt.UINT(0)
        user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), hdr_size)
        buf = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), hdr_size) != size.value:
            return
        hdr = RAWINPUTHEADER.from_buffer(buf)
        if hdr.dwType != RIM_TYPEHID:
            return
        handle = hdr.hDevice or 0
        device = self._device(handle)
        if device is None:
            return
        size_hid, count = struct.unpack_from("<II", buf, hdr_size)
        data = buf.raw[hdr_size + 8 : hdr_size + 8 + size_hid * count]
        for i in range(count):
            parsed = device.parse(data[i * size_hid : (i + 1) * size_hid])
            if parsed is None:
                continue
            frame = device.assembler.feed(*parsed)
            if frame is None:
                continue
            for c in frame:
                c.id = (handle & 0xFFFF) * 1000 + c.id  # unique across touchpads
                if device.has_pressure:
                    c.pressure = normalise_pressure(c.pressure, self._p_lo, self._p_hi)
            self._frames[handle] = frame
            if self._callback:
                self._callback([c for f in self._frames.values() for c in f])

    def _device(self, handle) -> _Device | None:
        if handle not in self._devices:
            try:
                self._devices[handle] = _Device(handle)
                self.pressure_supported = self.pressure_supported or self._devices[handle].has_pressure
                log.info("touchpad %s: %d finger slots", handle, len(self._devices[handle].fingers))
            except Exception:
                log.exception("unsupported HID touch device")
                self._devices[handle] = None
        return self._devices[handle]
