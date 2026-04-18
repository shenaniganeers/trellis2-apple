#!/usr/bin/env python3
"""
Suppress local convex "blob" protrusions on a repaired mesh relative to a source mesh.

The repaired mesh is treated as the target topology. We only modify local patches whose
vertices sit far from the source surface and protrude outward along the source normals.
Each patch is then fairing-smoothed with its boundary pinned, which approximates a
local area-minimizing surface without touching the rest of the mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locally suppress convex Poisson blobs.")
    parser.add_argument("source_mesh", type=Path, help="Original baseline/source mesh")
    parser.add_argument("repaired_mesh", type=Path, help="Closed/repaired mesh to fair locally")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--debug-mask-output", type=Path, default=None, help="Optional GLB with modified patches colored")
    parser.add_argument("--distance-threshold", type=float, default=0.012, help="Select repaired vertices farther than this from the source")
    parser.add_argument(
        "--selection-mode",
        choices=["outward", "allfar"],
        default="outward",
        help="How to select blob candidates: outward convex additions only, or any far unsupported patch",
    )
    parser.add_argument("--normal-dot-threshold", type=float, default=0.0, help="Require displacement dot source normal to exceed this")
    parser.add_argument("--min-patch-vertices", type=int, default=150, help="Discard outlier patches smaller than this")
    parser.add_argument("--dilate-rings", type=int, default=2, help="Grow selected patches by this many vertex-neighbor rings")
    parser.add_argument("--smooth-iterations", type=int, default=20, help="Fairing iterations per patch")
    parser.add_argument("--step-size", type=float, default=0.45, help="Jacobi fairing step size")
    parser.add_argument("--source-attract", type=float, default=0.15, help="Blend toward closest source points while fairing")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}")
    return mesh


def build_vertex_adjacency(mesh: trimesh.Trimesh) -> tuple[list[np.ndarray], coo_matrix]:
    edges = np.vstack(
        [
            mesh.faces[:, [0, 1]],
            mesh.faces[:, [1, 2]],
            mesh.faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    row = np.hstack([edges[:, 0], edges[:, 1]])
    col = np.hstack([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    graph = coo_matrix((data, (row, col)), shape=(len(mesh.vertices), len(mesh.vertices)))
    neighbors = [[] for _ in range(len(mesh.vertices))]
    for a, b in edges:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    neighbors = [np.asarray(n, dtype=np.int32) for n in neighbors]
    return neighbors, graph


def dilate_mask(mask: np.ndarray, neighbors: list[np.ndarray], rings: int) -> np.ndarray:
    if rings <= 0:
        return mask.copy()
    grown = mask.copy()
    frontier = np.nonzero(mask)[0].tolist()
    for _ in range(rings):
        new_frontier: list[int] = []
        for vid in frontier:
            nbrs = neighbors[vid]
            if len(nbrs) == 0:
                continue
            fresh = nbrs[~grown[nbrs]]
            if len(fresh):
                grown[fresh] = True
                new_frontier.extend(int(v) for v in fresh)
        frontier = new_frontier
        if not frontier:
            break
    return grown


def mask_components(mask: np.ndarray, graph: coo_matrix) -> tuple[int, np.ndarray]:
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return 0, np.empty(0, dtype=np.int32)
    sub = graph.tocsr()[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False, return_labels=True)
    full_labels = np.full(len(mask), -1, dtype=np.int32)
    full_labels[idx] = labels
    return n_comp, full_labels


def face_mask_from_vertices(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> np.ndarray:
    return vertex_mask[mesh.faces].any(axis=1)


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "body_count": int(mesh.body_count),
        "extents": [float(x) for x in mesh.extents.tolist()],
    }


def summarize_distances(dist: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(dist.mean()) if len(dist) else 0.0,
        "p95": float(np.quantile(dist, 0.95)) if len(dist) else 0.0,
        "max": float(dist.max()) if len(dist) else 0.0,
    }


def main() -> int:
    args = parse_args()
    source = load_mesh(args.source_mesh.resolve())
    repaired = load_mesh(args.repaired_mesh.resolve())

    prox = trimesh.proximity.ProximityQuery(source)
    closest, dist, triangle_id = prox.on_surface(repaired.vertices)
    source_normals = source.face_normals[triangle_id]
    displacement = repaired.vertices - closest
    normal_dot = np.einsum("ij,ij->i", displacement, source_normals)

    if args.selection_mode == "outward":
        core_mask = (dist > args.distance_threshold) & (normal_dot > args.normal_dot_threshold)
    else:
        core_mask = dist > args.distance_threshold
    neighbors, graph = build_vertex_adjacency(repaired)
    core_components, core_labels = mask_components(core_mask, graph)

    kept_core = np.zeros(len(repaired.vertices), dtype=bool)
    kept_patches: list[dict[str, object]] = []
    for comp_id in range(core_components):
        patch_vertices = core_labels == comp_id
        size = int(patch_vertices.sum())
        if size < args.min_patch_vertices:
            continue
        kept_core |= patch_vertices
        patch_dist = dist[patch_vertices]
        kept_patches.append(
            {
                "component": int(comp_id),
                "core_vertices": size,
                "distance": summarize_distances(patch_dist),
            }
        )

    grown_mask = dilate_mask(kept_core, neighbors, args.dilate_rings)
    grown_components, grown_labels = mask_components(grown_mask, graph)

    positions = repaired.vertices.copy()
    modified_vertices = np.zeros(len(repaired.vertices), dtype=bool)
    fairing_patches: list[dict[str, object]] = []

    for comp_id in range(grown_components):
        patch_mask = grown_labels == comp_id
        if not patch_mask.any():
            continue
        patch_idx = np.nonzero(patch_mask)[0]
        boundary_mask = np.zeros(len(patch_idx), dtype=bool)
        patch_set = patch_mask
        for local_i, vid in enumerate(patch_idx):
            nbrs = neighbors[vid]
            if len(nbrs) == 0 or np.any(~patch_set[nbrs]):
                boundary_mask[local_i] = True
        boundary_idx = patch_idx[boundary_mask]
        interior_idx = patch_idx[~boundary_mask]
        if len(interior_idx) == 0:
            continue

        modified_vertices[interior_idx] = True
        for _ in range(args.smooth_iterations):
            prev = positions.copy()
            for vid in interior_idx:
                nbrs = neighbors[vid]
                if len(nbrs) == 0:
                    continue
                lap_target = prev[nbrs].mean(axis=0)
                target = (1.0 - args.source_attract) * lap_target + args.source_attract * closest[vid]
                positions[vid] = prev[vid] + args.step_size * (target - prev[vid])

        fairing_patches.append(
            {
                "patch": int(comp_id),
                "grown_vertices": int(len(patch_idx)),
                "boundary_vertices": int(len(boundary_idx)),
                "interior_vertices": int(len(interior_idx)),
            }
        )

    out = trimesh.Trimesh(vertices=positions, faces=repaired.faces.copy(), process=False)
    out.export(args.output.resolve())

    if args.debug_mask_output is not None:
        face_mask = face_mask_from_vertices(repaired, modified_vertices)
        face_colors = np.full((len(repaired.faces), 4), [140, 140, 140, 255], dtype=np.uint8)
        face_colors[face_mask] = np.array([220, 38, 38, 255], dtype=np.uint8)
        debug = repaired.copy()
        debug.visual = trimesh.visual.ColorVisuals(mesh=debug, face_colors=face_colors)
        debug.export(args.debug_mask_output.resolve())

    after_closest, after_dist, _ = prox.on_surface(out.vertices)
    del after_closest

    report = {
        "source_mesh": str(args.source_mesh.resolve()),
        "repaired_mesh": str(args.repaired_mesh.resolve()),
        "output_mesh": str(args.output.resolve()),
        "distance_threshold": float(args.distance_threshold),
        "selection_mode": args.selection_mode,
        "normal_dot_threshold": float(args.normal_dot_threshold),
        "min_patch_vertices": int(args.min_patch_vertices),
        "dilate_rings": int(args.dilate_rings),
        "smooth_iterations": int(args.smooth_iterations),
        "step_size": float(args.step_size),
        "source_attract": float(args.source_attract),
        "source": mesh_stats(source),
        "repaired": mesh_stats(repaired),
        "output": mesh_stats(out),
        "initial_outlier_vertices": int(core_mask.sum()),
        "kept_outlier_vertices": int(kept_core.sum()),
        "modified_vertices": int(modified_vertices.sum()),
        "core_patch_count": int(core_components),
        "kept_core_patch_count": int(len(kept_patches)),
        "grown_patch_count": int(len(fairing_patches)),
        "distance_before": summarize_distances(dist),
        "distance_after": summarize_distances(after_dist),
        "modified_distance_before": summarize_distances(dist[modified_vertices]),
        "modified_distance_after": summarize_distances(after_dist[modified_vertices]),
        "kept_patches": kept_patches[:50],
        "fairing_patches": fairing_patches[:50],
        "debug_mask_output": str(args.debug_mask_output.resolve()) if args.debug_mask_output else None,
    }

    if args.report is not None:
        args.report.resolve().write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
