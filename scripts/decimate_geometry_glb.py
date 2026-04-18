#!/usr/bin/env python3
"""Decimate a geometry GLB before texturing/UV unwrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from cumesh import CuMesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input geometry GLB")
    parser.add_argument("output", type=Path, help="Output decimated GLB")
    parser.add_argument("--target-faces", type=int, required=True, help="Target triangle count")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report")
    parser.add_argument("--hole-fill-perimeter", type=float, default=0.03, help="Cleanup hole fill perimeter")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a Trimesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"{path} is empty")
    return mesh


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args.input)

    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64))

    worker = CuMesh()
    worker.init(vertices, faces)
    if args.verbose:
        print(f"Input: {worker.num_vertices} vertices, {worker.num_faces} faces")

    worker.simplify(int(args.target_faces), verbose=args.verbose)
    worker.remove_duplicate_faces()
    worker.repair_non_manifold_edges()
    worker.remove_small_connected_components(1e-5)
    if args.hole_fill_perimeter > 0:
        worker.fill_holes(max_hole_perimeter=args.hole_fill_perimeter)
    worker.unify_face_orientations()
    worker.compute_vertex_normals()

    out_vertices, out_faces = worker.read()
    out_normals = worker.read_vertex_normals()

    out_mesh = trimesh.Trimesh(
        vertices=out_vertices.cpu().numpy(),
        faces=out_faces.cpu().numpy(),
        vertex_normals=out_normals.cpu().numpy(),
        process=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(args.output)

    report = {
        "input": str(args.input),
        "target_faces": int(args.target_faces),
        "input_vertex_count": int(len(mesh.vertices)),
        "input_face_count": int(len(mesh.faces)),
        "output_vertex_count": int(len(out_mesh.vertices)),
        "output_face_count": int(len(out_mesh.faces)),
        "input_watertight": bool(mesh.is_watertight),
        "output_watertight": bool(out_mesh.is_watertight),
        "output_body_count": int(out_mesh.body_count),
        "hole_fill_perimeter": float(args.hole_fill_perimeter),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.verbose:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
