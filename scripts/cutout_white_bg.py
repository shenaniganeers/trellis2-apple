#!/usr/bin/env python3
"""Convert a near-white-background product render into an RGBA cutout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bg-threshold", type=int, default=248, help="Pixels brighter than this trend transparent")
    parser.add_argument("--fg-threshold", type=int, default=232, help="Pixels darker than this trend opaque")
    parser.add_argument("--trim", action="store_true", help="Crop to alpha bounds with a small margin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.input).convert("RGBA")
    arr = np.asarray(image, dtype=np.uint8)
    rgb = arr[..., :3].astype(np.float32)

    # Use the darkest channel as a conservative signal for "not background".
    signal = 255.0 - rgb.min(axis=-1)
    low = max(0.0, 255.0 - float(args.bg_threshold))
    high = max(low + 1.0, 255.0 - float(args.fg_threshold))
    alpha = np.clip((signal - low) / (high - low), 0.0, 1.0)
    alpha = (alpha * 255.0).astype(np.uint8)

    out = arr.copy()
    out[..., 3] = alpha
    result = Image.fromarray(out, mode="RGBA")

    if args.trim:
        ys, xs = np.nonzero(alpha > 0)
        if len(xs) and len(ys):
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = 16
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(out.shape[1] - 1, x1 + pad)
            y1 = min(out.shape[0] - 1, y1 + pad)
            result = result.crop((x0, y0, x1 + 1, y1 + 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
