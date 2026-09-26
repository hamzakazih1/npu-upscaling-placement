"""
Create the synthetic renderer proxy.

The 'renderer' in this experiment is a heavy GPU workload that reports its own
throughput. A real game would have better ecological validity but cannot be held
constant across runs, and cannot be instrumented without engine access. This
trade is a stated limitation, not a hidden one.

The model is a deep stack of convolutions: compute-heavy, memory-heavy, and
entirely standard operators, so every backend can run it.

Usage:
    python make_render_load.py --out render_load.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class RenderLoad(nn.Module):
    """A deliberately heavy convolutional stack standing in for a frame render."""

    def __init__(self, channels: int = 64, depth: int = 12):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers += [nn.Conv2d(channels, channels, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("render_load.onnx"))
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    args = parser.parse_args()

    model = RenderLoad(args.channels, args.depth).eval()
    dummy = torch.randn(1, 3, args.height, args.width)

    # Dynamic height and width so the same file serves both the 4K and 1080p
    # configurations. If a backend rejects dynamic axes, export two fixed files.
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["output"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"written {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {params:,} parameters)")
    print("If a provider rejects dynamic shapes, re-export at fixed sizes:")
    print("  python make_render_load.py --out load_4k.onnx   --height 1080 --width 1920")
    print("  python make_render_load.py --out load_1080.onnx --height 540  --width 960")


if __name__ == "__main__":
    main()
