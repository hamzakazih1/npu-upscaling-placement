import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep reshaped models and compiled contexts out of the user's cache."""
    monkeypatch.setenv("NPU_UPSCALE_CACHE", str(tmp_path / "cache"))
