"""
Register and select the QNN execution provider.

Two things about recent onnxruntime-qnn make this necessary, and both are easy
to miss because most documentation still describes the older behaviour.

1. QNN is a *plugin* execution provider. It does not appear in
   ort.get_available_providers() until its plugin library is registered
   explicitly, so a working NPU reports no QNN support out of the box.

2. Plugin providers are not selected through the providers=[...] argument.
   Passing "QNNExecutionProvider" there is silently ignored and the session
   falls back to CPU with no error. Selection happens by device, through
   SessionOptions.add_provider_for_devices().

The machine also exposes QNN twice, once for the NPU and once for the Adreno
GPU, so device type must be filtered as well as provider name.

Usage:
    from qnn_setup import ensure_qnn, make_qnn_session
    session = make_qnn_session("model_int8.onnx", device="npu")
"""

from __future__ import annotations

import os

import onnxruntime as ort

PLUGIN_DLL = "onnxruntime_providers_qnn.dll"
BACKENDS = {"npu": "QnnHtp.dll", "gpu": "QnnGpu.dll", "cpu": "QnnCpu.dll"}
_registered = False


def qnn_package_dir():
    """Directory of the installed onnxruntime-qnn package, if present."""
    try:
        import onnxruntime_qnn
    except ImportError:
        return None
    return os.path.dirname(onnxruntime_qnn.__file__)


def ensure_qnn(verbose: bool = True) -> bool:
    """
    Register the QNN plugin library. Safe to call repeatedly.

    Returns True if QNN devices are visible afterwards.
    """
    global _registered

    directory = qnn_package_dir()
    if directory is None:
        if verbose:
            print("onnxruntime-qnn not installed: pip install onnxruntime-qnn")
        return False

    if not _registered:
        plugin = os.path.join(directory, PLUGIN_DLL)
        if not os.path.exists(plugin):
            if verbose:
                print(f"QNN plugin library missing: {plugin}")
            return False
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", plugin)
            _registered = True
            if verbose:
                print(f"registered QNN plugin: {plugin}")
        except Exception as error:
            # Re-registering raises; treat that as already done.
            if "already" in str(error).lower():
                _registered = True
            else:
                if verbose:
                    print(f"QNN registration failed: {error}")
                return False

    return bool(qnn_devices())


def qnn_devices(device: str = "npu") -> list:
    """
    QNN devices of one hardware type.

    The provider is exposed once per backing device -- NPU and GPU both appear
    as QNNExecutionProvider -- so filtering by name alone would pick whichever
    comes first.
    """
    wanted = {
        "npu": ort.OrtHardwareDeviceType.NPU,
        "gpu": ort.OrtHardwareDeviceType.GPU,
        "cpu": ort.OrtHardwareDeviceType.CPU,
    }[device]

    try:
        devices = ort.get_ep_devices()
    except AttributeError:
        return []  # older runtime without the plugin device API

    return [d for d in devices
            if "QNN" in d.ep_name and d.device.type == wanted]


def make_qnn_session(model_path, device: str = "npu",
                     session_options=None,
                     performance_mode: str = "burst"):
    """
    Create a session bound to a QNN device.

    Raises if no matching device exists, rather than quietly producing a
    CPU-only session -- which is the failure mode this whole module exists to
    prevent.
    """
    if not ensure_qnn(verbose=False):
        raise RuntimeError("QNN is not available; run qnn_setup.py to diagnose")

    devices = qnn_devices(device)
    if not devices:
        raise RuntimeError(f"no QNN device of type {device!r} found")

    directory = qnn_package_dir()
    options = session_options or ort.SessionOptions()
    options.add_provider_for_devices(devices, {
        "backend_path": os.path.join(directory, BACKENDS[device]),
        "htp_performance_mode": performance_mode,
    })

    session = ort.InferenceSession(str(model_path), options)
    if "QNNExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "session created without QNN despite a matching device; "
            f"providers are {session.get_providers()}")
    return session


if __name__ == "__main__":
    ok = ensure_qnn()
    print("\nonnxruntime:", ort.__version__)
    print("built-in providers:", ort.get_available_providers())
    print("\nEP devices:")
    try:
        for d in ort.get_ep_devices():
            print(f"  {d.ep_name:<26} {d.device.type}  ({d.ep_vendor})")
    except AttributeError:
        print("  (this runtime has no plugin device API)")
    print(f"\nQNN NPU devices: {len(qnn_devices('npu'))}")
    print(f"QNN GPU devices: {len(qnn_devices('gpu'))}")
    print("\nQNN usable:", ok)
