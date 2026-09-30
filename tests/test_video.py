import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from npu_upscale import Upscaler
from npu_upscale.video import upscale_video


def test_video_round_trip(tmp_path):
    source = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    for i in range(6):
        writer.write(np.full((180, 320, 3), i * 40, dtype=np.uint8))
    writer.release()

    target = tmp_path / "clip_x2.mp4"
    result = upscale_video(source, target, Upscaler(device="cpu", frame_size=(320, 180)))
    assert result["frames"] == 6 and result["output"] == "640x360"

    capture = cv2.VideoCapture(str(target))
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (640, 360)
    capture.release()
