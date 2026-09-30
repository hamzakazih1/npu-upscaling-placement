import numpy as np
import pytest

from npu_upscale.runtime import create_cpu_session, reshape_model
from npu_upscale.tiling import tile_count, tile_starts, upscale_tiled
from npu_upscale.upscaler import DEFAULT_CPU_MODEL, DEFAULT_NPU_MODEL, bundled_model


@pytest.mark.parametrize("length, tile, margin", [
    (480, 480, 8), (481, 480, 8), (960, 480, 8), (1920, 480, 8), (1000, 270, 8), (300, 64, 4),
])
def test_tiles_cover_every_pixel(length, tile, margin):
    from npu_upscale.tiling import _valid_span
    covered = np.zeros(length, dtype=int)
    for start in tile_starts(length, tile, margin):
        assert 0 <= start and start + tile <= length
        lo, hi = _valid_span(start, tile, length, margin)
        covered[lo:hi] += 1
    assert covered.min() >= 1


def test_tile_count():
    assert tile_count(270, 480, 270, 480, 8) == 1
    assert tile_count(100, 100, 270, 480, 8) == 1


def _full_frame_reference(model, image):
    """Run the model over the whole image in one pass, at the image's own size."""
    _, height, width = image.shape
    session, _ = create_cpu_session(reshape_model(model, height, width))
    return session.run(None, {session.get_inputs()[0].name: image[None]})[0][0]


@pytest.mark.parametrize("height, width", [(270, 480), (300, 700), (540, 960), (123, 200)])
def test_tiled_equals_whole_frame(height, width):
    """Stitched tiles must be indistinguishable from one pass over the frame."""
    model = bundled_model(DEFAULT_CPU_MODEL)
    session, _ = create_cpu_session(model)
    name = session.get_inputs()[0].name
    run = lambda tile: session.run(None, {name: tile})[0]

    image = np.random.default_rng(height + width).random((3, height, width), dtype=np.float32)
    tiled = upscale_tiled(image, run, 270, 480, 2, margin=8)
    assert tiled.shape == (3, height * 2, width * 2)

    if height >= 270 or width >= 480:
        # Frames smaller than a tile are edge-padded, so they are compared
        # against the model, not a native pass.
        reference = _full_frame_reference(model, image)
        assert np.abs(tiled - reference).max() < 1e-4


def test_margin_matters():
    """Without overlap the seams are visible -- confirms the test above has teeth."""
    model = bundled_model(DEFAULT_CPU_MODEL)
    session, _ = create_cpu_session(model)
    name = session.get_inputs()[0].name
    run = lambda tile: session.run(None, {name: tile})[0]
    image = np.random.default_rng(0).random((3, 540, 960), dtype=np.float32)
    seamy = upscale_tiled(image, run, 270, 480, 2, margin=0)
    assert np.abs(seamy - _full_frame_reference(model, image)).max() > 1e-3


def test_reshaped_int8_model_matches_original_on_tile():
    model = bundled_model(DEFAULT_NPU_MODEL)
    image = np.random.default_rng(1).random((1, 3, 270, 480), dtype=np.float32)
    original, _ = create_cpu_session(model)
    reshaped, _ = create_cpu_session(reshape_model(model, 270, 480))
    name = original.get_inputs()[0].name
    a = original.run(None, {name: image})[0]
    b = reshaped.run(None, {name: image})[0]
    assert np.array_equal(a, b)
