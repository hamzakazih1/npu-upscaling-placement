"""
Create ONNX Runtime sessions that are guaranteed to run where they say.

This is the production version of benchmark/scripts/qnn_setup.py. It handles
the obstacles documented in the study, on any Snapdragon X-series chip:

  1. QNN as a plugin EP (onnxruntime-qnn >= 2.x) must be registered before it
     exists, and is selected by *device*, not through providers=[...].
  2. Older onnxruntime-qnn builds ship QNN as a built-in EP that *is* selected
     through providers=[...]. Both styles are supported; whichever the
     installed package provides is used.
  3. QNN is exposed once per backing device (NPU and Adreno GPU), so the
     device type is filtered, not just the provider name.
  4. The NPU is integer-only and ONNX Runtime silently runs unsupported nodes
     on the CPU. In strict mode (the default) CPU fallback is disabled at the
     session level, so a model that would not run wholly on the NPU fails to
     load instead of quietly running on the CPU under an NPU label.

It also caches the compiled QNN context per chip, so the multi-second graph
compilation on first load happens once per machine rather than every launch.
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import onnxruntime as ort

from .chips import Chip, detect_chip, emulation_warning

QNN = "QNNExecutionProvider"
CPU = "CPUExecutionProvider"

if sys.platform == "win32":
    PLUGIN_LIB = "onnxruntime_providers_qnn.dll"
    BACKENDS = {"npu": "QnnHtp.dll", "gpu": "QnnGpu.dll", "cpu": "QnnCpu.dll"}
else:
    PLUGIN_LIB = "libonnxruntime_providers_qnn.so"
    BACKENDS = {"npu": "libQnnHtp.so", "gpu": "libQnnGpu.so", "cpu": "libQnnCpu.so"}

PERFORMANCE_MODES = (
    "burst", "sustained_high_performance", "high_performance", "balanced",
    "low_balanced", "default", "power_saver", "low_power_saver",
    "high_power_saver", "extreme_power_saver",
)

_plugin_registered = False


class PlacementError(RuntimeError):
    """The session could not be placed on the requested device."""


@dataclass
class SessionInfo:
    """What a session actually runs on -- report this, not what was requested."""

    device: str                       # "npu" or "cpu"
    providers: list[str]
    chip: Chip
    model: Path
    qnn_style: str | None = None      # "plugin", "builtin" or None
    strict: bool = True
    context_cache: str | None = None  # "hit", "written" or None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        where = "Hexagon NPU via QNN" if self.device == "npu" else "CPU"
        text = f"{where} on {self.chip.describe()}"
        if self.device == "npu":
            text += "; CPU fallback " + ("disabled" if self.strict else "ALLOWED")
            if self.context_cache:
                text += f"; context cache {self.context_cache}"
        return text


# --------------------------------------------------------------------------
# Discovering QNN
# --------------------------------------------------------------------------

def qnn_package_dir() -> str | None:
    """Directory of the onnxruntime-qnn plugin package, if installed."""
    try:
        import onnxruntime_qnn  # type: ignore
    except ImportError:
        return None
    return os.path.dirname(onnxruntime_qnn.__file__)


def _register_plugin() -> bool:
    global _plugin_registered
    if _plugin_registered:
        return True
    directory = qnn_package_dir()
    register = getattr(ort, "register_execution_provider_library", None)
    if directory is None or register is None:
        return False
    library = os.path.join(directory, PLUGIN_LIB)
    if not os.path.exists(library):
        return False
    try:
        register(QNN, library)
    except Exception as error:  # re-registration raises; that is fine
        if "already" not in str(error).lower():
            return False
    _plugin_registered = True
    return True


def qnn_devices(device: str = "npu") -> list:
    """Plugin-EP QNN devices of one hardware type (empty on older runtimes)."""
    _register_plugin()
    try:
        wanted = {
            "npu": ort.OrtHardwareDeviceType.NPU,
            "gpu": ort.OrtHardwareDeviceType.GPU,
            "cpu": ort.OrtHardwareDeviceType.CPU,
        }[device]
        devices = ort.get_ep_devices()
    except AttributeError:
        return []
    return [d for d in devices if "QNN" in d.ep_name and d.device.type == wanted]


def qnn_style() -> str | None:
    """How QNN is exposed by the installed runtime: "plugin", "builtin" or None."""
    if qnn_devices("npu"):
        return "plugin"
    if QNN in ort.get_available_providers():
        return "builtin"
    return None


def _builtin_backend_path(device: str) -> str:
    """Absolute backend path for the built-in EP, or the bare name as a fallback."""
    name = BACKENDS[device]
    capi = Path(ort.__file__).parent / "capi"
    for candidate in (capi / name, Path(ort.__file__).parent / name):
        if candidate.exists():
            return str(candidate)
    return name


def npu_available() -> bool:
    return qnn_style() is not None


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def default_cache_dir() -> Path:
    base = os.environ.get("NPU_UPSCALE_CACHE")
    if base:
        return Path(base)
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(root) / "npu-upscale" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "npu-upscale"


def _context_path(model: Path, chip: Chip, cache_dir: Path) -> Path:
    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:12]
    runtime = ort.__version__.replace(".", "_")
    return cache_dir / f"{model.stem}_{digest}_{chip.slug}_ort{runtime}_ctx.onnx"


def _base_options(strict: bool, profile_prefix: str | None) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if strict:
        # Session creation fails if any node would land on the CPU EP.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    return options


def _qnn_session(model: Path, options: ort.SessionOptions, style: str,
                 performance_mode: str) -> ort.InferenceSession:
    qnn_options = {"htp_performance_mode": performance_mode}
    if style == "plugin":
        qnn_options["backend_path"] = os.path.join(qnn_package_dir(), BACKENDS["npu"])
        options.add_provider_for_devices(qnn_devices("npu"), qnn_options)
        session = ort.InferenceSession(str(model), options)
    else:
        qnn_options["backend_path"] = _builtin_backend_path("npu")
        session = ort.InferenceSession(str(model), options,
                                       providers=[(QNN, qnn_options)])
    if QNN not in session.get_providers():
        raise PlacementError(
            f"session was created without {QNN}; providers are {session.get_providers()}")
    return session


def create_npu_session(model: str | os.PathLike, *, strict: bool = True,
                       performance_mode: str = "burst", cache: bool = True,
                       cache_dir: str | os.PathLike | None = None,
                       profile_prefix: str | None = None,
                       chip: Chip | None = None) -> tuple[ort.InferenceSession, SessionInfo]:
    """
    A session on the Hexagon NPU, or PlacementError. Never a silent CPU session.
    """
    model = Path(model)
    chip = chip or detect_chip()
    if performance_mode not in PERFORMANCE_MODES:
        raise ValueError(f"performance_mode must be one of {PERFORMANCE_MODES}")

    problem = emulation_warning(chip)
    if problem:
        raise PlacementError(problem)

    style = qnn_style()
    if style is None:
        hint = ("pip install onnxruntime-qnn" if qnn_package_dir() is None
                else "onnxruntime-qnn is installed but exposes no NPU device; "
                     "update the Qualcomm NPU driver through Windows Update")
        raise PlacementError(f"no QNN NPU available on this machine ({hint})")

    info = SessionInfo(device="npu", providers=[], chip=chip, model=model,
                       qnn_style=style, strict=strict)

    context = None
    if cache and profile_prefix is None:
        context = _context_path(model, chip, Path(cache_dir or default_cache_dir()))
        if context.exists():
            try:
                options = _base_options(strict, None)
                session = _qnn_session(context, options, style, performance_mode)
                info.providers = session.get_providers()
                info.context_cache = "hit"
                return session, info
            except Exception as error:  # stale or foreign cache: rebuild
                info.notes.append(f"discarded context cache: {error}")
                try:
                    context.unlink()
                except OSError:
                    pass

    options = _base_options(strict, profile_prefix)
    if context is not None:
        try:
            context.parent.mkdir(parents=True, exist_ok=True)
            options.add_session_config_entry("ep.context_enable", "1")
            options.add_session_config_entry("ep.context_embed_mode", "1")
            options.add_session_config_entry("ep.context_file_path", str(context))
        except OSError as error:
            info.notes.append(f"context cache disabled: {error}")
            context = None

    try:
        session = _qnn_session(model, options, style, performance_mode)
    except PlacementError:
        raise
    except Exception as error:
        message = str(error)
        lowered = message.lower()
        if strict and ("fallback" in lowered or "not assigned" in lowered
                       or "cpuexecutionprovider" in lowered):
            raise PlacementError(
                "the model does not run entirely on the NPU, and CPU fallback is "
                "disabled. The Hexagon NPU needs a QDQ INT8 model with static input "
                f"shapes. Original error: {message}") from error
        raise PlacementError(f"QNN failed to create an NPU session: {message}") from error

    info.providers = session.get_providers()
    if context is not None and context.exists():
        info.context_cache = "written"
    return session, info


def create_cpu_session(model: str | os.PathLike, *, profile_prefix: str | None = None,
                       chip: Chip | None = None,
                       threads: int | None = None) -> tuple[ort.InferenceSession, SessionInfo]:
    model = Path(model)
    options = _base_options(strict=False, profile_prefix=profile_prefix)
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model), options, providers=[CPU])
    info = SessionInfo(device="cpu", providers=session.get_providers(),
                       chip=chip or detect_chip(), model=model, strict=False)
    return session, info


def create_session(npu_model: str | os.PathLike, cpu_model: str | os.PathLike | None = None,
                   *, device: str = "auto", **npu_kwargs) -> tuple[ort.InferenceSession, SessionInfo]:
    """
    device="npu"  NPU or PlacementError.
    device="cpu"  CPU.
    device="auto" NPU if it works, otherwise CPU -- with a warning, and the
                  returned SessionInfo says which one you got.
    """
    if device not in ("auto", "npu", "cpu"):
        raise ValueError("device must be 'auto', 'npu' or 'cpu'")
    cpu_model = cpu_model or npu_model

    if device == "cpu":
        return create_cpu_session(cpu_model, chip=npu_kwargs.get("chip"))
    try:
        return create_npu_session(npu_model, **npu_kwargs)
    except PlacementError as error:
        if device == "npu":
            raise
        warnings.warn(f"NPU unavailable, using CPU: {error}", RuntimeWarning, stacklevel=2)
        session, info = create_cpu_session(cpu_model, chip=npu_kwargs.get("chip"))
        info.notes.append(f"NPU unavailable: {error}")
        return session, info


# --------------------------------------------------------------------------
# Static shapes
# --------------------------------------------------------------------------

def reshape_model(model: str | os.PathLike, height: int, width: int,
                  cache_dir: str | os.PathLike | None = None) -> Path:
    """
    A copy of a fully convolutional model with a different static input size.

    The NPU needs static shapes, but nothing in Conv / DepthToSpace / Add /
    QDQ depends on the spatial size, so the bundled 270x480 model can be
    re-declared at the exact frame size. That removes tiling and its overlap
    (a 960x540 frame is 9 overlapping tiles, or one native pass). The result
    is cached next to the compiled contexts.
    """
    import onnx

    model = Path(model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:12]
    target = Path(cache_dir or default_cache_dir()) / f"{model.stem}_{digest}_{width}x{height}.onnx"
    if target.exists():
        return target

    graph_model = onnx.load(str(model))
    graph = graph_model.graph
    for node in graph.node:
        if node.op_type in ("Reshape", "Resize", "Upsample", "Flatten", "Gemm", "MatMul"):
            raise ValueError(f"{model.name} is not fully convolutional ({node.op_type}); "
                             "it cannot be re-declared at another size")

    source = graph.input[0].type.tensor_type.shape.dim
    sink = graph.output[0].type.tensor_type.shape.dim
    scale = sink[2].dim_value // source[2].dim_value
    source[2].dim_value, source[3].dim_value = height, width
    sink[2].dim_value, sink[3].dim_value = height * scale, width * scale
    del graph.value_info[:]  # stale intermediate shapes; re-inferred below
    graph_model = onnx.shape_inference.infer_shapes(graph_model)
    onnx.checker.check_model(graph_model)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    onnx.save(graph_model, str(temp))
    os.replace(temp, target)
    return target
