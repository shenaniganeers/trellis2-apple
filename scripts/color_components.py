#!/usr/bin/env python3
"""
Color a mesh by connected face components and export a debug GLB.

Useful when a visible "hole" is really a gap between disconnected shells or
an internal cavity rather than an open boundary loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = loaded.to_geometry()
        if isinstance(geometries, dict):
            meshes = [g for g in geometries.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(geometries, (list, tuple)):
            meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
        elif isinstance(geometries, trimesh.Trimesh):
            meshes = [geometries]
        else:
            meshes = []
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported geometry type: {type(loaded)}")
    return loaded


def palette(index: int) -> np.ndarray:
    colors = np.array(
        [
            [239, 68, 68, 255],
            [245, 158, 11, 255],
            [234, 179, 8, 255],
            [34, 197, 94, 255],
            [59, 130, 246, 255],
            [168, 85, 247, 255],
            [236, 72, 153, 255],
            [20, 184, 166, 255],
            [251, 113, 133, 255],
            [132, 204, 22, 255],
            [14, 165, 233, 255],
            [217, 70, 239, 255],
        ],
        dtype=np.uint8,
    )
    return colors[index % len(colors)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a mesh with face components colorized.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=50, help="Color the largest N components distinctly")
    parser.add_argument(
        "--top-band-y-min",
        type=float,
        default=None,
        help="Only color components whose bounds extend above this Y value",
    )
    parser.add_argument(
        "--base-gray",
        type=int,
        default=90,
        help="Gray level for non-highlighted components",
    )
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)

    comp_bounds = {}
    for comp_id in np.unique(labels):
        face_idx = np.nonzero(labels == comp_id)[0]
        verts_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        bounds = mesh.vertices[verts_idx].min(axis=0), mesh.vertices[verts_idx].max(axis=0)
        comp_bounds[int(comp_id)] = np.stack(bounds)

    ranked = sorted(range(len(counts)), key=lambda cid: int(counts[cid]), reverse=True)
    if args.top_band_y_min is not None:
        ranked = [cid for cid in ranked if comp_bounds[cid][1, 1] >= args.top_band_y_min]
    highlighted = ranked[: args.top]

    face_colors = np.full((len(mesh.faces), 4), [args.base_gray, args.base_gray, args.base_gray, 255], dtype=np.uint8)
    for i, comp_id in enumerate(highlighted):
        face_colors[labels == comp_id] = palette(i)

    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(mesh=colored, face_colors=face_colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    colored.export(args.output)

    print(f"mesh={args.mesh}")
    print(f"output={args.output}")
    print(f"components={len(counts)}")
    print(f"highlighted={len(highlighted)}")
    for i, comp_id in enumerate(highlighted[:20], start=1):
        bounds = comp_bounds[comp_id]
        center = bounds.mean(axis=0)
        size = bounds[1] - bounds[0]
        print(
            f"{i:>2}. component={comp_id} faces={int(counts[comp_id])} "
            f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
            f"size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})"
        )


if __name__ == "__main__":
    main()
