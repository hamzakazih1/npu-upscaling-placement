"""
Identify which Snapdragon X-series chip this machine has.

The study ran on one SKU (X Plus, X1P-42-100). Everything that makes the NPU
path work is shared across the family -- same Hexagon NPU generation within a
generation, same QNN HTP backend -- but the chip still matters for three
practical things:

  * a readable diagnosis ("X Elite, 45 TOPS NPU") instead of a CPU string,
  * the key for the compiled-context cache, which is only valid on the SoC
    that produced it,
  * spotting x64 Python running under emulation on an ARM64 Snapdragon,
    which cannot load QNN at all and is the most common setup failure.

Model numbers follow one scheme: X<gen><tier>-<bin>-<variant>, e.g.
X1E-80-100 (X Elite), X1P-42-100 (X Plus), X1-26-100 (Snapdragon X),
X2E-88-100 (X2 Elite). Windows reports them without dashes ("X1P42100"),
Linux device trees in lower case ("qcom,x1e80100"). All three are parsed.

Detection never guesses silently: an unrecognised machine is reported as
such, and NPU_UPSCALE_CHIP overrides detection when needed.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass, field

# Hexagon NPU generation per Snapdragon X generation. Headline INT8 TOPS are
# Qualcomm's published figures and are for reporting only; nothing is tuned
# on them.
GENERATIONS = {
    1: {"npu": "Hexagon (gen 1, HTP v73)", "npu_tops": 45},
    2: {"npu": "Hexagon (gen 2)", "npu_tops": 80},
}
TIERS = {"E": "Elite", "P": "Plus", "": ""}

# X<gen><tier>[-]<bin>[-]<variant>. The variant is usually 100, but dev kits
# use values such as 1DE (X1E-00-1DE).
_MODEL = re.compile(r"(?<![A-Z0-9])X([12])([EP]?)-?(\d{2})-?([0-9A-Z]{3})(?![0-9A-Z])",
                    re.IGNORECASE)
# Marketing name without a model number, e.g. "Snapdragon X Elite".
_MARKETING = re.compile(r"Snapdragon\S*\s+X(2)?\s*(Elite|Plus)?", re.IGNORECASE)


@dataclass(frozen=True)
class Chip:
    """A detected Snapdragon X-series SoC, or an unknown machine."""

    name: str                    # e.g. "Snapdragon X Plus"
    model: str | None = None     # e.g. "X1P-42-100"
    generation: int | None = None
    tier: str = ""               # "Elite", "Plus" or "" for plain Snapdragon X
    npu: str | None = None
    npu_tops: int | None = None
    source: str = "unknown"      # where the identification came from
    raw: str = field(default="", compare=False)

    @property
    def is_snapdragon_x(self) -> bool:
        return self.generation is not None

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, used to key the context cache."""
        if self.model:
            return self.model.lower()
        if self.generation:
            return f"x{self.generation}{self.tier[:1].lower()}-unknown"
        return "generic"

    def describe(self) -> str:
        if not self.is_snapdragon_x:
            return f"not a Snapdragon X-series chip ({self.raw or 'unidentified'})"
        text = self.name
        if self.model:
            text += f" ({self.model})"
        return f"{text}, {self.npu}, {self.npu_tops} TOPS NPU"


def _build(generation: int, tier_letter: str, model: str | None,
           source: str, raw: str) -> Chip:
    tier = TIERS[tier_letter.upper()]
    prefix = "Snapdragon X" if generation == 1 else f"Snapdragon X{generation}"
    name = f"{prefix} {tier}".strip()
    info = GENERATIONS[generation]
    return Chip(name=name, model=model, generation=generation, tier=tier,
                npu=info["npu"], npu_tops=info["npu_tops"],
                source=source, raw=raw)


def parse_chip(text: str, source: str = "string") -> Chip:
    """
    Identify a chip from any string that names it.

    >>> parse_chip("Snapdragon(R) X Plus - X1P42100 - Qualcomm(R) Oryon(TM) CPU").model
    'X1P-42-100'
    >>> parse_chip("qcom,x1e80100").name
    'Snapdragon X Elite'
    """
    raw = (text or "").strip()
    match = _MODEL.search(raw)
    if match:
        gen, tier, bin_, variant = match.groups()
        tier = tier.upper()
        model = f"X{gen}{tier}-{bin_}-{variant.upper()}"
        return _build(int(gen), tier, model, source, raw)

    match = _MARKETING.search(raw)
    if match:
        gen = int(match.group(1) or 1)
        tier = (match.group(2) or "")[:1].upper()
        return _build(gen, tier, None, source, raw)

    return Chip(name="unknown", source=source, raw=raw)


def _windows_processor_name() -> str | None:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value)
    except Exception:
        return None


def _linux_identifiers() -> list[tuple[str, str]]:
    found = []
    for path in ("/sys/firmware/devicetree/base/compatible",
                 "/proc/device-tree/compatible",
                 "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as handle:
                found.append((path, handle.read().replace(b"\0", b" ").decode(errors="ignore")))
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key.strip().lower() in ("model name", "hardware", "processor name"):
                    found.append(("/proc/cpuinfo", value.strip()))
                    break
    except OSError:
        pass
    return found


def detect_chip() -> Chip:
    """Detect the chip in this machine. NPU_UPSCALE_CHIP overrides detection."""
    override = os.environ.get("NPU_UPSCALE_CHIP")
    if override:
        return parse_chip(override, source="NPU_UPSCALE_CHIP")

    candidates: list[tuple[str, str]] = []
    if sys.platform == "win32":
        name = _windows_processor_name()
        if name:
            candidates.append(("registry ProcessorNameString", name))
        identifier = os.environ.get("PROCESSOR_IDENTIFIER")
        if identifier:
            candidates.append(("PROCESSOR_IDENTIFIER", identifier))
    elif sys.platform.startswith("linux"):
        candidates.extend(_linux_identifiers())
    candidates.append(("platform.processor()", platform.processor()))

    for source, text in candidates:
        chip = parse_chip(text, source=source)
        if chip.is_snapdragon_x:
            return chip
    first = next((text for _, text in candidates if text.strip()), "")
    return Chip(name="unknown", source="none matched", raw=first.strip()[:120])


def os_is_arm64() -> bool:
    """True if the operating system is ARM64, even when Python is emulated x64."""
    if sys.platform == "win32":
        # Under x64 emulation PROCESSOR_ARCHITECTURE says AMD64, but the native
        # architecture leaks through PROCESSOR_ARCHITEW6432 or the registry.
        for var in ("PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"):
            if os.environ.get(var, "").upper() == "ARM64":
                return True
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            value, _ = winreg.QueryValueEx(key, "PROCESSOR_ARCHITECTURE")
            return str(value).upper() == "ARM64"
        except Exception:
            return False
    return platform.machine().lower() in ("arm64", "aarch64")


def python_is_arm64() -> bool:
    return platform.machine().lower() in ("arm64", "aarch64")


def emulation_warning(chip: Chip | None = None) -> str | None:
    """A message if this Python cannot reach the NPU because it is emulated."""
    chip = chip or detect_chip()
    if (chip.is_snapdragon_x or os_is_arm64()) and not python_is_arm64():
        return (f"Python is {platform.machine()} running under emulation on an ARM64 "
                "machine. QNN cannot load in an emulated process: install the ARM64 "
                "build of Python from python.org and recreate the virtual environment.")
    return None
