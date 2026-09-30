import warnings

import numpy as np
import pytest

from npu_upscale import PlacementError, Upscaler, npu_available
from npu_upscale.runtime import create_npu_session
from npu_upscale.upscaler import DEFAULT_NPU_MODEL, bundled_model

no_npu = pytest.mark.skipif(npu_available(), reason="this machine has a working NPU")
needs_npu = pytest.mark.skipif(not npu_available(), reason="needs a Snapdragon NPU with QNN")


@pytest.fixture(scope="module")
def cpu_upscaler():
    return Upscaler(device="cpu")


def test_rgb(cpu_upscaler):
    image = np.random.default_rng(0).integers(0, 256, (300, 500, 3), dtype=np.uint8)
    out = cpu_upscaler.upscale(image)
    assert out.shape == (600, 1000, 3) and out.dtype == np.uint8


def test_rgba_and_gray(cpu_upscaler):
    rgba = np.random.default_rng(0).integers(0, 256, (50, 60, 4), dtype=np.uint8)
    out = cpu_upscaler.upscale(rgba)
    assert out.shape == (100, 120, 4)
    assert np.array_equal(out[::2, ::2, 3], rgba[..., 3])
    gray = rgba[..., 0].copy()
    assert cpu_upscaler.upscale(gray).shape == (100, 120)


def test_beats_nearest_neighbour_on_smooth_content(cpu_upscaler):
    """A sanity check that the model upscales rather than merely resizes."""
    y, x = np.mgrid[0:540, 0:960] / 60.0
    hr = np.stack([np.sin(x) * 0.4 + 0.5, np.cos(y) * 0.4 + 0.5, np.sin(x + y) * 0.4 + 0.5])
    hr = (hr * 255).astype(np.uint8).transpose(1, 2, 0)
    lr = hr.reshape(270, 2, 480, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    nearest = lr.repeat(2, 0).repeat(2, 1)
    model = cpu_upscaler.upscale(lr)
    err = lambda a: np.abs(a.astype(float) - hr).mean()
    assert err(model) < err(nearest)


def test_stream_preserves_order(cpu_upscaler):
    frames = [np.full((20, 30, 3), i * 20, dtype=np.uint8) for i in range(8)]
    outs = list(cpu_upscaler.stream(frames, workers=3, prefetch=2))
    assert len(outs) == 8
    means = [int(round(o.mean())) for o in outs]
    assert means == sorted(means)


def test_native_frame_size_matches_tiling(cpu_upscaler):
    image = np.random.default_rng(3).integers(0, 256, (540, 960, 3), dtype=np.uint8)
    native = Upscaler(device="cpu", frame_size=(960, 540))
    assert native.tiles_for(540, 960) == 1 and cpu_upscaler.tiles_for(540, 960) > 1
    diff = np.abs(native.upscale(image).astype(int) - cpu_upscaler.upscale(image).astype(int))
    assert diff.max() <= 1


def test_benchmark_reports_budget(cpu_upscaler):
    result = cpu_upscaler.benchmark(270, 480, seconds=0.1, warmup=1)
    assert result["device"] == "cpu" and result["tiles"] == 1
    assert result["budget_60fps"] == pytest.approx(result["budget_30fps"] * 2, rel=0.01)


@no_npu
def test_npu_request_never_silently_becomes_cpu():
    with pytest.raises(PlacementError):
        Upscaler(device="npu")


@no_npu
def test_auto_falls_back_loudly_and_reports_it():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        up = Upscaler(device="auto")
    assert up.info.device == "cpu"
    assert any("NPU unavailable" in str(w.message) for w in caught)


@needs_npu
def test_npu_runs_whole_model_on_hexagon():
    from npu_upscale.placement import verify_placement
    report = verify_placement(bundled_model(DEFAULT_NPU_MODEL), device="npu")
    assert report["fully_on_device"], report
    assert report["max_abs_diff_vs_cpu"] < 0.05


@needs_npu
def test_strict_rejects_fp32_model_on_npu():
    """Obstacle 4 in the study: fp32 would run on the CPU under an NPU label."""
    from npu_upscale.upscaler import DEFAULT_CPU_MODEL
    with pytest.raises(PlacementError):
        create_npu_session(bundled_model(DEFAULT_CPU_MODEL), cache=False)


@needs_npu
def test_context_cache_round_trip():
    model = bundled_model(DEFAULT_NPU_MODEL)
    _, first = create_npu_session(model)
    _, second = create_npu_session(model)
    assert second.context_cache in ("hit", None)
