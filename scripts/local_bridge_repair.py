#!/usr/bin/env python3
"""
Conservative local component-bridging experiment.

This keeps the source mesh unchanged and only adds new faces that zipper
between nearby boundary loops belonging to different connected components.
It is intended as a geometry-only debug/repair pass before any future
rebake/decimation integration.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class BoundaryLoop:
    loop_id: int
    component_id: int
    vertices: np.ndarray
    points: np.ndarray
    closed: bool
    tree: cKDTree
    bounds: np.ndarray


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


def connected_components(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency,
        node_count=len(mesh.faces),
    )
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    ranked = np.argsort(counts)[::-1]
    return labels, counts, ranked


def trace_boundary_loops(boundary_edges: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for u, v in boundary_edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    visited_edges: set[tuple[int, int]] = set()
    loops: list[tuple[np.ndarray, bool]] = []

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    starts = [v for v, nbrs in adjacency.items() if len(nbrs) != 2]
    starts.extend(v for v, nbrs in adjacency.items() if len(nbrs) == 2)

    for start in starts:
        for first in adjacency[start]:
            key = edge_key(start, first)
            if key in visited_edges:
                continue

            chain = [start]
            prev = start
            curr = first
            visited_edges.add(key)
            chain.append(curr)

            closed = False
            while True:
                nbrs = adjacency[curr]
                next_candidates = [n for n in nbrs if n != prev and edge_key(curr, n) not in visited_edges]
                if not next_candidates:
                    if len(nbrs) > 1 and chain[0] in nbrs and edge_key(curr, chain[0]) not in visited_edges:
                        visited_edges.add(edge_key(curr, chain[0]))
                        closed = True
                    break
                nxt = next_candidates[0]
                visited_edges.add(edge_key(curr, nxt))
                prev, curr = curr, nxt
                chain.append(curr)
                if curr == chain[0]:
                    closed = True
                    break

            if closed and len(chain) > 1 and chain[-1] == chain[0]:
                chain = chain[:-1]
            if len(chain) >= 3:
                loops.append((np.asarray(chain, dtype=np.int64), closed))
    return loops


def component_boundary_loops(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    counts: np.ndarray,
    ranked_components: np.ndarray,
    top_components: int,
    min_loop_vertices: int,
    min_component_faces: int,
) -> list[BoundaryLoop]:
    loops: list[BoundaryLoop] = []
    loop_id = 0
    for component_id in ranked_components[:top_components]:
        if int(counts[component_id]) < min_component_faces:
            continue
        face_idx = np.nonzero(labels == component_id)[0]
        if len(face_idx) == 0:
            continue
        faces = mesh.faces[face_idx]
        edges = np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[edge_counts == 1]
        if len(boundary_edges) == 0:
            continue
        for vertex_ids, closed in trace_boundary_loops(boundary_edges):
            if len(vertex_ids) < min_loop_vertices:
                continue
            points = mesh.vertices[vertex_ids]
            bounds = np.stack([points.min(axis=0), points.max(axis=0)])
            loops.append(
                BoundaryLoop(
                    loop_id=loop_id,
                    component_id=int(component_id),
                    vertices=vertex_ids,
                    points=points,
                    closed=closed,
                    tree=cKDTree(points),
                    bounds=bounds,
                )
            )
            loop_id += 1
    return loops


def bbox_distance(a: np.ndarray, b: np.ndarray) -> float:
    gap = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(gap))


def extract_chain(loop: BoundaryLoop, seed_index: int, half_window: int) -> np.ndarray:
    verts = loop.vertices
    n = len(verts)
    if loop.closed:
        window = min(half_window, max((n - 1) // 2, 1))
        idx = [(seed_index + offset) % n for offset in range(-window, window + 1)]
        return verts[np.asarray(idx, dtype=np.int64)]
    start = max(0, seed_index - half_window)
    end = min(n, seed_index + half_window + 1)
    return verts[start:end]


def choose_orientation(mesh: trimesh.Trimesh, chain_a: np.ndarray, chain_b: np.ndarray) -> np.ndarray:
    a_pts = mesh.vertices[chain_a]
    b_pts = mesh.vertices[chain_b]
    forward = np.linalg.norm(a_pts[0] - b_pts[0]) + np.linalg.norm(a_pts[-1] - b_pts[-1])
    reverse = np.linalg.norm(a_pts[0] - b_pts[-1]) + np.linalg.norm(a_pts[-1] - b_pts[0])
    return chain_b[::-1] if reverse < forward else chain_b


def zipper_faces(
    mesh: trimesh.Trimesh,
    chain_a: np.ndarray,
    chain_b: np.ndarray,
    max_edge_length: float,
    min_area: float,
) -> list[list[int]]:
    pts = mesh.vertices
    faces: list[list[int]] = []
    i = 0
    j = 0
    while i < len(chain_a) - 1 or j < len(chain_b) - 1:
        if i == len(chain_a) - 1:
            tri = [int(chain_a[i]), int(chain_b[j]), int(chain_b[j + 1])]
            j += 1
        elif j == len(chain_b) - 1:
            tri = [int(chain_a[i]), int(chain_b[j]), int(chain_a[i + 1])]
            i += 1
        else:
            cost_a = np.linalg.norm(pts[chain_a[i + 1]] - pts[chain_b[j]])
            cost_b = np.linalg.norm(pts[chain_a[i]] - pts[chain_b[j + 1]])
            if cost_a <= cost_b:
                tri = [int(chain_a[i]), int(chain_b[j]), int(chain_a[i + 1])]
                i += 1
            else:
                tri = [int(chain_a[i]), int(chain_b[j]), int(chain_b[j + 1])]
                j += 1

        tri_pts = pts[np.asarray(tri)]
        edge_lengths = [
            np.linalg.norm(tri_pts[0] - tri_pts[1]),
            np.linalg.norm(tri_pts[1] - tri_pts[2]),
            np.linalg.norm(tri_pts[2] - tri_pts[0]),
        ]
        area = 0.5 * np.linalg.norm(np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0]))
        if max(edge_lengths) <= max_edge_length and area >= min_area:
            faces.append(tri)
    return faces


def main() -> int:
    parser = argparse.ArgumentParser(description="Locally bridge nearby boundary loops between components.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-components", type=int, default=40)
    parser.add_argument("--min-component-faces", type=int, default=1000)
    parser.add_argument("--min-loop-vertices", type=int, default=12)
    parser.add_argument("--min-gap", type=float, default=1e-4)
    parser.add_argument("--max-gap", type=float, default=0.018)
    parser.add_argument("--half-window", type=int, default=10)
    parser.add_argument("--max-bridges", type=int, default=16)
    parser.add_argument("--max-edge-factor", type=float, default=4.0)
    parser.add_argument("--min-triangle-area", type=float, default=1e-7)
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    labels, counts, ranked_components = connected_components(mesh)
    loops = component_boundary_loops(
        mesh=mesh,
        labels=labels,
        counts=counts,
        ranked_components=ranked_components,
        top_components=args.top_components,
        min_loop_vertices=args.min_loop_vertices,
        min_component_faces=args.min_component_faces,
    )

    candidates = []
    for i, loop_a in enumerate(loops):
        for loop_b in loops[i + 1 :]:
            if loop_a.component_id == loop_b.component_id:
                continue
            if bbox_distance(loop_a.bounds, loop_b.bounds) > args.max_gap:
                continue
            dist, idx = loop_b.tree.query(loop_a.points, k=1)
            seed_a = int(np.argmin(dist))
            gap = float(dist[seed_a])
            if gap < args.min_gap or gap > args.max_gap:
                continue
            seed_b = int(idx[seed_a])
            candidates.append(
                {
                    "gap": gap,
                    "loop_a": loop_a,
                    "loop_b": loop_b,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                }
            )

    candidates.sort(key=lambda item: item["gap"])
    used_loops: set[int] = set()
    added_faces: list[list[int]] = []
    applied = []

    for candidate in candidates:
        loop_a = candidate["loop_a"]
        loop_b = candidate["loop_b"]
        if loop_a.loop_id in used_loops or loop_b.loop_id in used_loops:
            continue
        chain_a = extract_chain(loop_a, candidate["seed_a"], args.half_window)
        chain_b = extract_chain(loop_b, candidate["seed_b"], args.half_window)
        if len(chain_a) < 3 or len(chain_b) < 3:
            continue
        chain_b = choose_orientation(mesh, chain_a, chain_b)
        max_edge = max(args.max_gap * args.max_edge_factor, candidate["gap"] * args.max_edge_factor)
        bridge = zipper_faces(mesh, chain_a, chain_b, max_edge_length=max_edge, min_area=args.min_triangle_area)
        if not bridge:
            continue
        added_faces.extend(bridge)
        used_loops.add(loop_a.loop_id)
        used_loops.add(loop_b.loop_id)
        applied.append(
            {
                "gap": candidate["gap"],
                "component_a": loop_a.component_id,
                "component_b": loop_b.component_id,
                "loop_a": loop_a.loop_id,
                "loop_b": loop_b.loop_id,
                "chain_a_vertices": int(len(chain_a)),
                "chain_b_vertices": int(len(chain_b)),
                "bridge_faces": int(len(bridge)),
            }
        )
        if len(applied) >= args.max_bridges:
            break

    combined_faces = mesh.faces.copy()
    if added_faces:
        combined_faces = np.vstack([combined_faces, np.asarray(added_faces, dtype=np.int64)])

    # Remove exact duplicate faces while preserving the original vertices.
    sorted_faces = np.sort(combined_faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    combined_faces = combined_faces[unique_idx]

    repaired = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=combined_faces, process=False)
    trimesh.repair.fix_normals(repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(args.output)

    report = {
        "input_mesh": str(args.mesh.resolve()),
        "output_mesh": str(args.output.resolve()),
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "source_body_count": int(mesh.body_count),
        "source_watertight": bool(mesh.is_watertight),
        "candidate_pairs": int(len(candidates)),
        "applied_bridges": len(applied),
        "added_faces": int(len(combined_faces) - len(mesh.faces)),
        "result_vertices": int(len(repaired.vertices)),
        "result_faces": int(len(repaired.faces)),
        "result_body_count": int(repaired.body_count),
        "result_watertight": bool(repaired.is_watertight),
        "bridges": applied,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
