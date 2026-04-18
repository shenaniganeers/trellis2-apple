#!/usr/bin/env python3
"""
Plane-guided local repair for unsupported hard-surface patches.

The goal is to keep the repaired shell topology, but replace a remaining unsupported
patch with a flatter, more planar local solution based on nearby support planes from
the source mesh. This is meant for hard-surface cleanup after volumetric reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from plane_patch_debug import (
    build_vertex_adjacency,
    cluster_normals,
    color_mesh,
    dilate_mask,
    face_mask_from_vertices,
    fit_plane,
    load_mesh,
    mask_components,
    submesh_from_face_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plane-guided repair for hard-surface patches.")
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("repaired_mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--debug-mask-output", type=Path, default=None)
    parser.add_argument("--distance-threshold", type=float, default=0.006)
    parser.add_argument("--selection-mode", choices=["outward", "allfar"], default="allfar")
    parser.add_argument("--normal-dot-threshold", type=float, default=0.0)
    parser.add_argument("--min-patch-vertices", type=int, default=40)
    parser.add_argument("--patch-rank", type=int, default=0)
    parser.add_argument("--support-rings", type=int, default=6)
    parser.add_argument("--repair-rings", type=int, default=4)
    parser.add_argument("--max-planes", type=int, default=3)
    parser.add_argument("--max-kept-planes", type=int, default=2)
    parser.add_argument(
        "--use-raw-plane-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional raw plane indices to use directly instead of automatic dedup selection",
    )
    parser.add_argument("--normal-angle-deg", type=float, default=20.0)
    parser.add_argument("--merge-angle-deg", type=float, default=15.0)
    parser.add_argument("--min-plane-samples", type=int, default=20)
    parser.add_argument("--plane-pad", type=float, default=0.015)
    parser.add_argument("--smooth-iterations", type=int, default=80)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--core-plane-weight", type=float, default=0.9)
    parser.add_argument("--grown-plane-weight", type=float, default=0.45)
    parser.add_argument("--final-core-project", type=float, default=0.9)
    parser.add_argument("--seam-smooth-iterations", type=int, default=0, help="Extra Laplacian smoothing iterations on the grown transition ring")
    parser.add_argument("--seam-step-size", type=float, default=0.25, help="Step size for seam-only smoothing")
    return parser.parse_args()


def summarize_distances(dist: np.ndarray) -> dict[str, float]:
    if len(dist) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(dist.mean()),
        "p95": float(np.quantile(dist, 0.95)),
        "max": float(dist.max()),
    }


def plane_score(plane: dict[str, object]) -> float:
    p95 = float(plane["fit_error_p95"])
    max_err = float(plane["fit_error_max"])
    samples = float(plane["sample_count"])
    return p95 + 0.25 * max_err - 1e-6 * samples


def dedupe_planes(planes: list[dict[str, object]], merge_angle_deg: float, max_kept: int) -> list[dict[str, object]]:
    if not planes:
        return []
    cos_thresh = float(np.cos(np.deg2rad(merge_angle_deg)))
    ordered = sorted(planes, key=plane_score)
    kept: list[dict[str, object]] = []
    for plane in ordered:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        if any(abs(float(np.dot(normal, np.asarray(other["normal"], dtype=np.float64)))) >= cos_thresh for other in kept):
            continue
        kept.append(plane)
        if len(kept) >= max_kept:
            break
    return kept


def project_to_plane(points: np.ndarray, plane: dict[str, object]) -> np.ndarray:
    normal = np.asarray(plane["normal"], dtype=np.float64)
    centroid = np.asarray(plane["centroid"], dtype=np.float64)
    signed = (points - centroid) @ normal
    return points - signed[:, None] * normal[None, :]


def select_patch(
    source: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    neighbors: list[np.ndarray],
    graph,
    distance_threshold: float,
    selection_mode: str,
    normal_dot_threshold: float,
    min_patch_vertices: int,
    patch_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    prox = trimesh.proximity.ProximityQuery(source)
    closest, dist, triangle_id = prox.on_surface(repaired.vertices)
    source_normals = source.face_normals[triangle_id]
    displacement = repaired.vertices - closest
    normal_dot = np.einsum("ij,ij->i", displacement, source_normals)
    if selection_mode == "outward":
        core_mask = (dist > distance_threshold) & (normal_dot > normal_dot_threshold)
    else:
        core_mask = dist > distance_threshold
    core_components, core_labels = mask_components(core_mask, graph)
    ranked: list[tuple[int, int]] = []
    for comp_id in range(core_components):
        size = int((core_labels == comp_id).sum())
        if size >= min_patch_vertices:
            ranked.append((comp_id, size))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked:
        raise SystemExit("No unsupported patches passed the selection threshold")
    if patch_rank < 0 or patch_rank >= len(ranked):
        raise SystemExit(f"patch-rank {patch_rank} out of range, only {len(ranked)} supported patches")
    patch_comp, patch_size = ranked[patch_rank]
    patch_mask = core_labels == patch_comp
    return closest, dist, source_normals, normal_dot, core_mask, {
        "patch_mask": patch_mask,
        "patch_component": int(patch_comp),
        "patch_vertices": int(patch_size),
        "candidate_patches": [{"component": int(cid), "vertices": int(sz)} for cid, sz in ranked],
    }


def main() -> int:
    args = parse_args()
    source = load_mesh(args.source_mesh.resolve())
    repaired = load_mesh(args.repaired_mesh.resolve())
    neighbors, graph = build_vertex_adjacency(repaired)

    closest, dist, source_normals, _, _, patch_info = select_patch(
        source=source,
        repaired=repaired,
        neighbors=neighbors,
        graph=graph,
        distance_threshold=args.distance_threshold,
        selection_mode=args.selection_mode,
        normal_dot_threshold=args.normal_dot_threshold,
        min_patch_vertices=args.min_patch_vertices,
        patch_rank=args.patch_rank,
    )
    patch_mask = patch_info["patch_mask"]

    support_region = dilate_mask(patch_mask, neighbors, args.support_rings)
    support_mask = support_region & ~patch_mask
    support_vertices = np.nonzero(support_mask)[0]
    support_points = closest[support_vertices]
    support_normals = source_normals[support_vertices]

    clusters = cluster_normals(
        normals=support_normals,
        max_planes=args.max_planes,
        angle_deg=args.normal_angle_deg,
        min_samples=args.min_plane_samples,
    )
    raw_planes: list[dict[str, object]] = []
    for i, cluster_idx in enumerate(clusters):
        pts = support_points[cluster_idx]
        nrms = support_normals[cluster_idx]
        plane = fit_plane(pts, nrms, pad=args.plane_pad)
        raw_planes.append(
            {
                "plane_index": i,
                "sample_count": int(len(cluster_idx)),
                "centroid": [float(x) for x in plane["centroid"]],
                "normal": [float(x) for x in plane["normal"]],
                "fit_error_mean": float(plane["fit_error_mean"]),
                "fit_error_p95": float(plane["fit_error_p95"]),
                "fit_error_max": float(plane["fit_error_max"]),
            }
        )

    if args.use_raw_plane_indices:
        wanted = {int(i) for i in args.use_raw_plane_indices}
        planes = [plane for plane in raw_planes if int(plane["plane_index"]) in wanted]
        planes.sort(key=lambda plane: int(plane["plane_index"]))
    else:
        planes = dedupe_planes(raw_planes, merge_angle_deg=args.merge_angle_deg, max_kept=args.max_kept_planes)
    if not planes:
        raise SystemExit("No support planes remained after deduplication")

    repair_region = dilate_mask(patch_mask, neighbors, args.repair_rings)
    patch_or_grown = repair_region.copy()
    patch_vertices = np.nonzero(patch_mask)[0]
    region_vertices = np.nonzero(repair_region)[0]

    boundary_mask = np.zeros(len(region_vertices), dtype=bool)
    for local_i, vid in enumerate(region_vertices):
        nbrs = neighbors[vid]
        if len(nbrs) == 0 or np.any(~repair_region[nbrs]):
            boundary_mask[local_i] = True
    boundary_idx = region_vertices[boundary_mask]
    interior_idx = region_vertices[~boundary_mask]

    positions = repaired.vertices.copy()
    plane_centroids = np.asarray([plane["centroid"] for plane in planes], dtype=np.float64)
    plane_normals = np.asarray([plane["normal"] for plane in planes], dtype=np.float64)

    def nearest_plane_ids(points: np.ndarray) -> np.ndarray:
        signed = np.abs((points[:, None, :] - plane_centroids[None, :, :]) * plane_normals[None, :, :]).sum(axis=2)
        return np.argmin(signed, axis=1)

    assignment = np.full(len(repaired.vertices), -1, dtype=np.int32)
    if len(interior_idx):
        assignment[interior_idx] = nearest_plane_ids(positions[interior_idx])

    grown_only_mask = repair_region & ~patch_mask
    for _ in range(args.smooth_iterations):
        prev = positions.copy()
        if len(interior_idx):
            assignment[interior_idx] = nearest_plane_ids(prev[interior_idx])
        for vid in interior_idx:
            nbrs = neighbors[vid]
            if len(nbrs) == 0:
                continue
            lap_target = prev[nbrs].mean(axis=0)
            plane = planes[int(assignment[vid])]
            plane_proj = project_to_plane(prev[vid : vid + 1], plane)[0]
            plane_weight = args.core_plane_weight if patch_mask[vid] else args.grown_plane_weight
            target = (1.0 - plane_weight) * lap_target + plane_weight * plane_proj
            positions[vid] = prev[vid] + args.step_size * (target - prev[vid])

    if args.final_core_project > 0 and len(patch_vertices):
        core_ids = nearest_plane_ids(positions[patch_vertices])
        projected = positions[patch_vertices].copy()
        for i, vid in enumerate(patch_vertices):
            projected[i] = project_to_plane(positions[vid : vid + 1], planes[int(core_ids[i])])[0]
        positions[patch_vertices] = (
            (1.0 - args.final_core_project) * positions[patch_vertices]
            + args.final_core_project * projected
        )

    seam_vertices = np.nonzero(grown_only_mask)[0]
    if args.seam_smooth_iterations > 0 and len(seam_vertices):
        for _ in range(args.seam_smooth_iterations):
            prev = positions.copy()
            for vid in seam_vertices:
                nbrs = neighbors[vid]
                if len(nbrs) == 0:
                    continue
                lap_target = prev[nbrs].mean(axis=0)
                positions[vid] = prev[vid] + args.seam_step_size * (lap_target - prev[vid])

        if args.final_core_project > 0 and len(patch_vertices):
            core_ids = nearest_plane_ids(positions[patch_vertices])
            projected = positions[patch_vertices].copy()
            for i, vid in enumerate(patch_vertices):
                projected[i] = project_to_plane(positions[vid : vid + 1], planes[int(core_ids[i])])[0]
            positions[patch_vertices] = (
                (1.0 - args.final_core_project) * positions[patch_vertices]
                + args.final_core_project * projected
            )

    out = trimesh.Trimesh(vertices=positions, faces=repaired.faces.copy(), process=False)
    out.export(args.output.resolve())

    if args.debug_mask_output is not None:
        face_colors = np.full((len(repaired.faces), 4), [150, 150, 150, 255], dtype=np.uint8)
        patch_faces = face_mask_from_vertices(repaired, patch_mask)
        grown_faces = face_mask_from_vertices(repaired, grown_only_mask)
        face_colors[grown_faces] = np.array([245, 158, 11, 255], dtype=np.uint8)
        face_colors[patch_faces] = np.array([220, 38, 38, 255], dtype=np.uint8)
        debug = repaired.copy()
        debug.visual = trimesh.visual.ColorVisuals(mesh=debug, face_colors=face_colors)
        debug.export(args.debug_mask_output.resolve())

    prox = trimesh.proximity.ProximityQuery(source)
    _, after_dist, _ = prox.on_surface(out.vertices)

    report = {
        "source_mesh": str(args.source_mesh.resolve()),
        "repaired_mesh": str(args.repaired_mesh.resolve()),
        "output_mesh": str(args.output.resolve()),
        "distance_threshold": float(args.distance_threshold),
        "selection_mode": args.selection_mode,
        "patch_component": patch_info["patch_component"],
        "patch_vertices": patch_info["patch_vertices"],
        "support_vertices": int(len(support_vertices)),
        "repair_region_vertices": int(repair_region.sum()),
        "boundary_vertices": int(len(boundary_idx)),
        "interior_vertices": int(len(interior_idx)),
        "kept_planes": planes,
        "raw_planes": raw_planes,
        "candidate_patches": patch_info["candidate_patches"],
        "distance_before": summarize_distances(dist),
        "distance_after": summarize_distances(after_dist),
        "patch_distance_before": summarize_distances(dist[patch_mask]),
        "patch_distance_after": summarize_distances(after_dist[patch_mask]),
        "grown_distance_before": summarize_distances(dist[grown_only_mask]),
        "grown_distance_after": summarize_distances(after_dist[grown_only_mask]),
        "seam_vertices": int(len(seam_vertices)),
        "seam_smooth_iterations": int(args.seam_smooth_iterations),
        "seam_step_size": float(args.seam_step_size),
        "output_watertight": bool(out.is_watertight),
        "output_body_count": int(out.body_count),
        "debug_mask_output": str(args.debug_mask_output.resolve()) if args.debug_mask_output else None,
    }

    if args.report is not None:
        args.report.resolve().write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
