#!/usr/bin/env python3
"""
Run one Trellis2 MLX generation in a fresh process and exit.

This is the safer path for long/high-memory jobs on Apple Silicon because the
process lifetime is bounded to a single generation and export.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import o_voxel
import torch
from PIL import Image

from mlx_backend.pipeline import create_mlx_pipeline


def free_memory() -> None:
    gc.collect()
    try:
        mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


@contextmanager
def single_run_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"Another TRELLIS job is already active: {lock_path}")
    try:
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Trellis2 MLX job and exit.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output GLB path")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "TRELLIS.2-4B")
    parser.add_argument("--resolution", choices=["512", "1024"], default="1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--tex-steps", type=int, default=12)
    parser.add_argument("--ss-guidance", type=float, default=7.5)
    parser.add_argument("--shape-guidance", type=float, default=7.5)
    parser.add_argument("--tex-guidance", type=float, default=1.0)
    parser.add_argument("--target-faces", type=int, default=250000)
    parser.add_argument("--texture-size", type=int, default=2048, choices=[1024, 2048, 4096])
    parser.add_argument("--hole-fill-perimeter", type=float, default=0.03)
    parser.add_argument("--close-shell", action="store_true")
    parser.add_argument("--close-shell-resolution", type=int, default=192)
    parser.add_argument("--close-shell-iters", type=int, default=1)
    parser.add_argument("--close-shell-project-back", type=float, default=1.0)
    parser.add_argument("--force-opaque", action="store_true")
    parser.add_argument("--skip-background-removal", action="store_true")
    parser.add_argument("--remesh", action="store_true")
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.9)
    parser.add_argument("--lock-file", default=os.path.join(tempfile.gettempdir(), "trellis2-mlx.lock"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"Input image not found: {args.image}", file=sys.stderr)
        return 2

    pipeline_type = {"512": "512", "1024": "1024_cascade"}[args.resolution]

    with single_run_lock(args.lock_file):
        image = Image.open(args.image)
        pipeline = create_mlx_pipeline(weights_path=str(args.weights))
        try:
            meshes = pipeline.run(
                image,
                seed=args.seed,
                pipeline_type=pipeline_type,
                skip_background_removal=args.skip_background_removal,
                sparse_structure_sampler_params={"steps": args.ss_steps, "guidance_strength": args.ss_guidance},
                shape_slat_sampler_params={"steps": args.shape_steps, "guidance_strength": args.shape_guidance},
                tex_slat_sampler_params={"steps": args.tex_steps, "guidance_strength": args.tex_guidance},
            )
            mesh = meshes[0]
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.target_faces,
                texture_size=args.texture_size,
                hole_fill_max_perimeter=args.hole_fill_perimeter,
                close_shell=args.close_shell,
                close_shell_resolution=args.close_shell_resolution,
                close_shell_iters=args.close_shell_iters,
                close_shell_project_back=args.close_shell_project_back,
                force_opaque=args.force_opaque,
                remesh=args.remesh,
                remesh_band=args.remesh_band,
                remesh_project=args.remesh_project,
                verbose=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            glb.export(args.output)
            print(args.output)
        finally:
            for name in ("glb", "mesh", "meshes", "pipeline", "image"):
                if name in locals():
                    del locals()[name]
            free_memory()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
