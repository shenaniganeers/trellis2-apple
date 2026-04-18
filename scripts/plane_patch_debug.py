#!/usr/bin/env python3
"""
Inspect an unsupported repaired-mesh patch and fit neighboring support planes.

This is a debug tool for hard-surface repair. It identifies a large "far from source"
patch on the repaired shell, gathers nearby support samples from the source mesh,
fits dominant planes, and exports an overlay GLB showing:
- the repaired shell in gray
- the selected unsupported patch in red
- the neighboring support ring in blue
- fitted plane quads in distinct translucent colors
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
    parser = argparse.ArgumentParser(description="Fit support planes around an unsupported patch.")
    parser.add_argument("source_mesh", type=Path, help="Reference/source mesh")
    parser.add_argument("repaired_mesh", type=Path, help="Repaired/closed mesh")
    parser.add_argument("--output", type=Path, required=True, help="Overlay GLB output")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report")
    parser.add_argument("--distance-threshold", type=float, default=0.01, help="Unsupported patch threshold")
    parser.add_argument("--selection-mode", choices=["outward", "allfar"], default="allfar")
    parser.add_argument("--normal-dot-threshold", type=float, default=0.0)
    parser.add_argument("--min-patch-vertices", type=int, default=40)
    parser.add_argument("--patch-rank", type=int, default=0, help="0 selects the largest unsupported patch")
    parser.add_argument("--support-rings", type=int, default=4, help="Neighbor rings outside the patch used for support sampling")
    parser.add_argument("--max-planes", type=int, default=3)
    parser.add_argument("--normal-angle-deg", type=float, default=20.0, help="Normal clustering threshold in degrees")
    parser.add_argument("--min-plane-samples", type=int, default=30, help="Discard plane clusters smaller than this")
    parser.add_argument("--plane-pad", type=float, default=0.01, help="Extra pad added around fitted plane quads")
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
    grown = mask.copy()
    frontier = np.nonzero(mask)[0].tolist()
    for _ in range(max(rings, 0)):
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


def submesh_from_face_mask(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    face_index = np.nonzero(face_mask)[0]
    if len(face_index) == 0:
        raise ValueError("No faces selected")
    sub = mesh.submesh([face_index], append=True, repair=False)
    if isinstance(sub, list):
        sub = trimesh.util.concatenate(sub)
    return trimesh.Trimesh(vertices=sub.vertices.copy(), faces=sub.faces.copy(), process=False)


def color_mesh(mesh: trimesh.Trimesh, rgba: list[int]) -> trimesh.Trimesh:
    colored = mesh.copy()
    face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(colored.faces), 1))
    colored.visual = trimesh.visual.ColorVisuals(mesh=colored, face_colors=face_colors)
    return colored


def unsigned_normal_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a @ b.T)


def cluster_normals(normals: np.ndarray, max_planes: int, angle_deg: float, min_samples: int) -> list[np.ndarray]:
    if len(normals) == 0:
        return []
    threshold = float(np.cos(np.deg2rad(angle_deg)))
    unassigned = np.ones(len(normals), dtype=bool)
    clusters: list[np.ndarray] = []
    while unassigned.any() and len(clusters) < max_planes:
        active = np.nonzero(unassigned)[0]
        sims = unsigned_normal_similarity(normals[active], normals[active])
        cover_counts = (sims >= threshold).sum(axis=1)
        seed_local = int(np.argmax(cover_counts))
        seed_global = active[seed_local]
        seed_sim = np.abs(normals @ normals[seed_global])
        cluster = unassigned & (seed_sim >= threshold)
        if int(cluster.sum()) < min_samples:
            unassigned[seed_global] = False
            continue
        clusters.append(np.nonzero(cluster)[0])
        unassigned[cluster] = False
    return clusters


def fit_plane(points: np.ndarray, normals: np.ndarray, pad: float) -> dict[str, object]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis_u = vh[0]
    basis_v = vh[1]
    normal = vh[2]
    avg_normal = normals.mean(axis=0)
    if np.dot(normal, avg_normal) < 0:
        normal = -normal
    proj_u = centered @ basis_u
    proj_v = centered @ basis_v
    u_min, u_max = float(proj_u.min() - pad), float(proj_u.max() + pad)
    v_min, v_max = float(proj_v.min() - pad), float(proj_v.max() + pad)
    corners = np.array(
        [
            centroid + u_min * basis_u + v_min * basis_v,
            centroid + u_max * basis_u + v_min * basis_v,
            centroid + u_max * basis_u + v_max * basis_v,
            centroid + u_min * basis_u + v_max * basis_v,
        ],
        dtype=np.float64,
    )
    dist = np.abs(centered @ normal)
    return {
        "centroid": centroid,
        "normal": normal,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "corners": corners,
        "fit_error_mean": float(dist.mean()),
        "fit_error_p95": float(np.quantile(dist, 0.95)),
        "fit_error_max": float(dist.max()),
        "size_u": float(u_max - u_min),
        "size_v": float(v_max - v_min),
    }


def plane_quad_mesh(corners: np.ndarray) -> trimesh.Trimesh:
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=corners.copy(), faces=faces, process=False)


def summarize_distances(dist: np.ndarray) -> dict[str, float]:
    if len(dist) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(dist.mean()),
        "p95": float(np.quantile(dist, 0.95)),
        "max": float(dist.max()),
    }


def main() -> int:
    args = parse_args()
    source = load_mesh(args.source_mesh.resolve())
    repaired = load_mesh(args.repaired_mesh.resolve())
    neighbors, graph = build_vertex_adjacency(repaired)

    prox = trimesh.proximity.ProximityQuery(source)
    closest, dist, triangle_id = prox.on_surface(repaired.vertices)
    source_normals = source.face_normals[triangle_id]
    displacement = repaired.vertices - closest
    normal_dot = np.einsum("ij,ij->i", displacement, source_normals)

    if args.selection_mode == "outward":
        core_mask = (dist > args.distance_threshold) & (normal_dot > args.normal_dot_threshold)
    else:
        core_mask = dist > args.distance_threshold

    core_components, core_labels = mask_components(core_mask, graph)
    ranked: list[tuple[int, int]] = []
    for comp_id in range(core_components):
        size = int((core_labels == comp_id).sum())
        if size >= args.min_patch_vertices:
            ranked.append((comp_id, size))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked:
        raise SystemExit("No unsupported patches passed the selection threshold")
    if args.patch_rank < 0 or args.patch_rank >= len(ranked):
        raise SystemExit(f"patch-rank {args.patch_rank} out of range, only {len(ranked)} supported patches")

    patch_comp, patch_size = ranked[args.patch_rank]
    patch_mask = core_labels == patch_comp
    patch_vertices = np.nonzero(patch_mask)[0]

    grown = dilate_mask(patch_mask, neighbors, args.support_rings)
    support_mask = grown & ~patch_mask
    support_vertices = np.nonzero(support_mask)[0]
    support_points = closest[support_vertices]
    support_normals = source_normals[support_vertices]

    clusters = cluster_normals(
        normals=support_normals,
        max_planes=args.max_planes,
        angle_deg=args.normal_angle_deg,
        min_samples=args.min_plane_samples,
    )

    plane_colors = [
        [59, 130, 246, 140],
        [34, 197, 94, 140],
        [245, 158, 11, 140],
        [168, 85, 247, 140],
    ]
    planes: list[dict[str, object]] = []
    scene = trimesh.Scene()
    scene.add_geometry(color_mesh(repaired, [160, 160, 160, 255]), geom_name="repaired")
    scene.add_geometry(
        color_mesh(submesh_from_face_mask(repaired, face_mask_from_vertices(repaired, patch_mask)), [220, 38, 38, 255]),
        geom_name="patch",
    )
    if len(support_vertices):
        scene.add_geometry(
            color_mesh(submesh_from_face_mask(repaired, face_mask_from_vertices(repaired, support_mask)), [37, 99, 235, 180]),
            geom_name="support_ring",
        )

    for i, cluster_idx in enumerate(clusters):
        pts = support_points[cluster_idx]
        nrms = support_normals[cluster_idx]
        plane = fit_plane(pts, nrms, pad=args.plane_pad)
        plane_mesh = plane_quad_mesh(plane["corners"])
        plane_mesh = color_mesh(plane_mesh, plane_colors[i % len(plane_colors)])
        scene.add_geometry(plane_mesh, geom_name=f"plane_{i}")
        planes.append(
            {
                "plane_index": i,
                "sample_count": int(len(cluster_idx)),
                "centroid": [float(x) for x in plane["centroid"]],
                "normal": [float(x) for x in plane["normal"]],
                "size_u": float(plane["size_u"]),
                "size_v": float(plane["size_v"]),
                "fit_error_mean": float(plane["fit_error_mean"]),
                "fit_error_p95": float(plane["fit_error_p95"]),
                "fit_error_max": float(plane["fit_error_max"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output.resolve())

    report = {
        "source_mesh": str(args.source_mesh.resolve()),
        "repaired_mesh": str(args.repaired_mesh.resolve()),
        "output": str(args.output.resolve()),
        "distance_threshold": float(args.distance_threshold),
        "selection_mode": args.selection_mode,
        "normal_dot_threshold": float(args.normal_dot_threshold),
        "patch_rank": int(args.patch_rank),
        "patch_component": int(patch_comp),
        "patch_vertices": int(patch_size),
        "patch_distance": summarize_distances(dist[patch_mask]),
        "support_vertices": int(len(support_vertices)),
        "support_rings": int(args.support_rings),
        "support_distance": summarize_distances(dist[support_mask]),
        "support_clusters_considered": int(len(clusters)),
        "planes": planes,
        "candidate_patches": [
            {"component": int(comp_id), "vertices": int(size)}
            for comp_id, size in ranked
        ],
    }

    if args.report is not None:
        args.report.resolve().write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
