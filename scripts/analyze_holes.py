#!/usr/bin/env python3
"""
Analyze open boundary loops in a mesh/GLB and rank them by perimeter.

The perimeter units match the mesh coordinate system used during export.
For TRELLIS outputs this is typically the normalized object-space scale,
so the reported numbers are directly comparable to `hole_fill_max_perimeter`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    counts = np.bincount(
        mesh.faces_unique_edges.reshape(-1),
        minlength=len(mesh.edges_unique),
    )
    return mesh.edges_unique[counts == 1]


def edge_length(vertices: np.ndarray, edge: tuple[int, int]) -> float:
    a, b = edge
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def extract_boundary_paths(mesh: trimesh.Trimesh) -> list[dict]:
    vertices = mesh.vertices
    edges = boundary_edges(mesh)
    adjacency: dict[int, list[int]] = defaultdict(list)
    edge_lengths: dict[tuple[int, int], float] = {}

    for a, b in edges:
        a_i = int(a)
        b_i = int(b)
        adjacency[a_i].append(b_i)
        adjacency[b_i].append(a_i)
        edge_lengths[canonical_edge(a_i, b_i)] = edge_length(vertices, (a_i, b_i))

    unused = set(edge_lengths.keys())
    paths: list[dict] = []

    while unused:
        seed = next(iter(unused))
        a, b = seed
        start = a if len(adjacency[a]) != 2 else (b if len(adjacency[b]) != 2 else a)

        ordered = [start]
        prev = None
        current = start
        perimeter = 0.0
        used_here = 0
        closed = False

        while True:
            candidates = [
                nxt for nxt in adjacency[current]
                if canonical_edge(current, nxt) in unused and nxt != prev
            ]
            if not candidates:
                break

            nxt = candidates[0]
            edge = canonical_edge(current, nxt)
            unused.remove(edge)
            perimeter += edge_lengths[edge]
            used_here += 1
            ordered.append(nxt)
            prev, current = current, nxt

            if current == start:
                closed = True
                break

        path_vertices = ordered[:-1] if closed else ordered
        coords = vertices[np.array(path_vertices)]
        centroid = coords.mean(axis=0).tolist() if len(coords) else [0.0, 0.0, 0.0]
        bounds = (
            [coords.min(axis=0).tolist(), coords.max(axis=0).tolist()]
            if len(coords)
            else [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        paths.append(
            {
                "kind": "loop" if closed else "open_boundary",
                "perimeter": perimeter,
                "edge_count": used_here,
                "vertex_count": len(coords),
                "centroid": centroid,
                "bounds": bounds,
                "vertex_indices": path_vertices,
            }
        )

    return sorted(paths, key=lambda item: item["perimeter"], reverse=True)


def summarize(paths: list[dict], threshold: float | None) -> dict:
    loops = [p for p in paths if p["kind"] == "loop"]
    open_paths = [p for p in paths if p["kind"] != "loop"]
    summary = {
        "total_boundary_paths": len(paths),
        "closed_loops": len(loops),
        "open_boundary_paths": len(open_paths),
    }
    if threshold is not None:
        summary["loops_at_or_below_threshold"] = sum(p["perimeter"] <= threshold for p in loops)
        summary["loops_above_threshold"] = sum(p["perimeter"] > threshold for p in loops)
    return summary


def loop_palette(index: int) -> np.ndarray:
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
        ],
        dtype=np.uint8,
    )
    return colors[index % len(colors)]


def build_loop_overlay_scene(
    mesh: trimesh.Trimesh,
    paths: list[dict],
    top: int,
    min_perimeter: float = 0.0,
    max_perimeter: float | None = None,
    tube_radius: float = 0.002,
    sections: int = 6,
    include_mesh: bool = True,
) -> trimesh.Scene:
    scene = trimesh.Scene()
    if include_mesh:
        scene.add_geometry(mesh.copy(), node_name="source_mesh")

    vertices = mesh.vertices
    selected = select_paths(paths, top=top, min_perimeter=min_perimeter, max_perimeter=max_perimeter)

    for idx, path in enumerate(selected):
        loop_vertices = path["vertex_indices"]
        if len(loop_vertices) < 2:
            continue

        segments = []
        for i in range(len(loop_vertices)):
            a = vertices[loop_vertices[i]]
            b = vertices[loop_vertices[(i + 1) % len(loop_vertices)]]
            if np.allclose(a, b):
                continue
            seg = trimesh.creation.cylinder(
                radius=tube_radius,
                segment=np.stack([a, b]),
                sections=sections,
            )
            seg.visual.face_colors = loop_palette(idx)
            segments.append(seg)

        if not segments:
            continue

        scene.add_geometry(
            trimesh.util.concatenate(segments),
            node_name=f"hole_loop_{idx + 1:02d}",
            geom_name=f"hole_loop_{idx + 1:02d}_p{path['perimeter']:.4f}",
        )

    return scene


def select_paths(
    paths: list[dict],
    top: int,
    min_perimeter: float = 0.0,
    max_perimeter: float | None = None,
) -> list[dict]:
    return [
        p for p in paths
        if p["kind"] == "loop"
        and p["perimeter"] >= min_perimeter
        and (max_perimeter is None or p["perimeter"] <= max_perimeter)
    ][:top]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank mesh boundary holes by perimeter.")
    parser.add_argument("mesh", type=Path, help="Path to mesh or GLB file")
    parser.add_argument("--top", type=int, default=20, help="Number of largest holes to print")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Compare perimeters against a hole-fill threshold",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--export-loop-glb",
        type=Path,
        default=None,
        help="Write an overlay GLB with the top boundary loops rendered as tubes",
    )
    parser.add_argument(
        "--min-perimeter",
        type=float,
        default=0.0,
        help="Only include loops at or above this perimeter in the overlay export",
    )
    parser.add_argument(
        "--max-perimeter",
        type=float,
        default=None,
        help="Only include loops at or below this perimeter in the overlay export",
    )
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.002,
        help="Tube radius for exported loop overlay",
    )
    parser.add_argument(
        "--no-source-mesh",
        action="store_true",
        help="Export only the loop tubes, without the original mesh",
    )
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    paths = extract_boundary_paths(mesh)
    summary = summarize(paths, args.threshold)
    selected_paths = select_paths(
        paths,
        top=args.top,
        min_perimeter=args.min_perimeter,
        max_perimeter=args.max_perimeter,
    )

    if args.export_loop_glb is not None:
        scene = build_loop_overlay_scene(
            mesh=mesh,
            paths=paths,
            top=args.top,
            min_perimeter=args.min_perimeter,
            max_perimeter=args.max_perimeter,
            tube_radius=args.tube_radius,
            include_mesh=not args.no_source_mesh,
        )
        args.export_loop_glb.parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.export_loop_glb)

    if args.json:
        payload = {
            "mesh": str(args.mesh),
            "summary": summary,
            "selected_paths": selected_paths,
            "export_loop_glb": str(args.export_loop_glb) if args.export_loop_glb else None,
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Mesh: {args.mesh}")
    print(f"Vertices: {len(mesh.vertices):,}")
    print(f"Faces: {len(mesh.faces):,}")
    print(f"Watertight: {mesh.is_watertight}")
    print(
        "Boundary paths: {total_boundary_paths} total | "
        "{closed_loops} loops | {open_boundary_paths} open".format(**summary)
    )
    if args.threshold is not None:
        print(
            f"Threshold {args.threshold:.6f}: "
            f"{summary['loops_at_or_below_threshold']} loops <= threshold | "
            f"{summary['loops_above_threshold']} loops > threshold"
        )
    if args.export_loop_glb is not None:
        print(f"Overlay GLB: {args.export_loop_glb}")
    if args.min_perimeter > 0 or args.max_perimeter is not None:
        max_label = "inf" if args.max_perimeter is None else f"{args.max_perimeter:.6f}"
        print(f"Selected band: {args.min_perimeter:.6f} <= perimeter <= {max_label}")
        print(f"Selected loops exported/listed: {len(selected_paths)}")
    print()

    for idx, path in enumerate(selected_paths, start=1):
        centroid = ", ".join(f"{v:.4f}" for v in path["centroid"])
        print(
            f"{idx:>2}. {path['kind']:>13} | "
            f"perimeter={path['perimeter']:.6f} | "
            f"edges={path['edge_count']:>4} | "
            f"centroid=({centroid})"
        )


if __name__ == "__main__":
    main()
