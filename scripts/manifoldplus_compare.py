#!/usr/bin/env python3
"""
Run ManifoldPlus on a mesh and compare the result to the source geometry.

This is an experiment harness, not a production pipeline stage. The upstream
ManifoldPlus project is fetched and built separately because its license is
non-commercial.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare ManifoldPlus repair.")
    parser.add_argument("input_mesh", type=Path, help="Input mesh path (.obj/.off/.glb/.gltf)")
    parser.add_argument("--manifold-bin", type=Path, default=Path("/Users/jjfeiler/Developer/_external/ManifoldPlus/build/manifold"))
    parser.add_argument("--depth", type=int, default=8, help="ManifoldPlus octree depth")
    parser.add_argument("--samples", type=int, default=100000, help="Surface samples for approximate distance stats")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Output prefix; defaults next to input mesh")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single mesh from {path}")
    return mesh


def approximate_surface_distances(a: trimesh.Trimesh, b: trimesh.Trimesh, samples: int) -> dict[str, float]:
    a_pts, _ = trimesh.sample.sample_surface(a, samples)
    b_pts, _ = trimesh.sample.sample_surface(b, samples)
    a_tree = cKDTree(a_pts)
    b_tree = cKDTree(b_pts)
    b_to_a = a_tree.query(b_pts, k=1)[0]
    a_to_b = b_tree.query(a_pts, k=1)[0]
    return {
        "repaired_to_source_mean": float(b_to_a.mean()),
        "repaired_to_source_p95": float(np.quantile(b_to_a, 0.95)),
        "repaired_to_source_max": float(b_to_a.max()),
        "source_to_repaired_mean": float(a_to_b.mean()),
        "source_to_repaired_p95": float(np.quantile(a_to_b, 0.95)),
        "source_to_repaired_max": float(a_to_b.max()),
    }


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "body_count": int(mesh.body_count),
        "extents": [float(x) for x in mesh.extents.tolist()],
    }


def main() -> int:
    args = parse_args()
    input_mesh = args.input_mesh.resolve()
    if not input_mesh.exists():
        raise SystemExit(f"Input mesh not found: {input_mesh}")
    manifold_bin = args.manifold_bin.resolve()
    if not manifold_bin.exists():
        raise SystemExit(f"ManifoldPlus binary not found: {manifold_bin}")

    if args.output_prefix is None:
        stem = input_mesh.with_suffix("")
        output_prefix = stem.parent / f"{stem.name}_manifoldplus_d{args.depth}"
    else:
        output_prefix = args.output_prefix.resolve()

    source_obj = output_prefix.with_name(f"{output_prefix.name}_source.obj")
    repaired_obj = output_prefix.with_suffix(".obj")
    repaired_glb = output_prefix.with_suffix(".glb")
    report_json = output_prefix.with_suffix(".json")

    source = load_mesh(input_mesh)
    source.export(source_obj)

    subprocess.run(
        [
            str(manifold_bin),
            "--input",
            str(source_obj),
            "--output",
            str(repaired_obj),
            "--depth",
            str(args.depth),
        ],
        check=True,
    )

    repaired = load_mesh(repaired_obj)
    repaired.export(repaired_glb)

    report = {
        "input_mesh": str(input_mesh),
        "source_obj": str(source_obj),
        "repaired_obj": str(repaired_obj),
        "repaired_glb": str(repaired_glb),
        "depth": args.depth,
        "source": mesh_stats(source),
        "repaired": mesh_stats(repaired),
        "approx_surface_distance": approximate_surface_distances(source, repaired, args.samples),
    }
    report_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
