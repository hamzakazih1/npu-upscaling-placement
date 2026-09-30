import numpy as np
from PIL import Image

from npu_upscale.cli import main


def test_image_file_and_folder(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    for i in range(2):
        Image.fromarray(np.full((40, 50, 3), i * 100, dtype=np.uint8)).save(source / f"f{i}.png")

    assert main(["image", str(source / "f0.png"), "-o", str(tmp_path / "one.png"),
                 "--device", "cpu"]) == 0
    assert Image.open(tmp_path / "one.png").size == (100, 80)

    out = tmp_path / "out"
    assert main(["image", str(source), "-o", str(out), "--device", "cpu"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["f0_x2.png", "f1_x2.png"]


def test_bench_json(capsys):
    assert main(["bench", "--size", "480x270", "--seconds", "0.1", "--device", "cpu", "--json"]) == 0
    assert '"tiles": 1' in capsys.readouterr().out
