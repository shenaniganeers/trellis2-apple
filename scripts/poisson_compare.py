#!/usr/bin/env python3
"""
Run Screened Poisson reconstruction on a mesh and compare the result to the source.

This is an experiment harness for geometry repair. It produces raw and projected
outputs so the effect of projection-back can be inspected separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import pymeshlab as ml
except Exception:
    ml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Screened Poisson repair and compare.")
    parser.add_argument("input_mesh", type=Path, help="Input mesh path")
    parser.add_argument("--backend", choices=["pymeshlab", "open3d"], default="pymeshlab", help="Poisson backend")
    parser.add_argument("--prune-min-faces", type=int, default=0, help="If >0, drop source connected components smaller than this many faces before Poisson")
    parser.add_argument("--depth", type=int, default=8, help="Poisson octree depth")
    parser.add_argument("--sample-count", type=int, default=400000, help="Surface samples with normals")
    parser.add_argument("--normal-knn", type=int, default=50, help="Neighborhood size for normal orientation")
    parser.add_argument("--density-quantile", type=float, default=0.01, help="Drop lowest-density vertices by quantile")
    parser.add_argument("--scale", type=float, default=1.05, help="Poisson reconstruction scale")
    parser.add_argument("--linear-fit", action="store_true", help="Use linear fit in Poisson reconstruction")
    parser.add_argument("--samples-per-node", type=float, default=1.5, help="MeshLab samples per node")
    parser.add_argument("--point-weight", type=float, default=4.0, help="MeshLab interpolation weight")
    parser.add_argument("--poisson-iters", type=int, default=8, help="MeshLab solver iterations")
    parser.add_argument("--threads", type=int, default=1, help="MeshLab screened-Poisson thread count")
    parser.add_argument("--prefill-holes-max-size", type=int, default=0, help="If >0, close small holes in the source mesh before Poisson")
    parser.add_argument("--prefill-selfintersection", action="store_true", help="Allow self-intersecting prefill patches in MeshLab hole closing")
    parser.add_argument("--crop-padding-frac", type=float, default=0.02, help="AABB crop padding as a fraction of max extent")
    parser.add_argument("--project-back-blend", type=float, default=1.0, help="Blend factor toward closest source point")
    parser.add_argument("--project-back-max-distance", type=float, default=0.01, help="Only project vertices within this source distance")
    parser.add_argument("--distance-samples", type=int, default=100000, help="Surface samples for approximate distance stats")
    parser.add_argument("--keep-largest-component", action="store_true", help="Also export largest raw/projected reconstructed component")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Output prefix; defaults next to input mesh")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single mesh from {path}")
    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "body_count": int(mesh.body_count),
        "extents": [float(x) for x in mesh.extents.tolist()],
    }


def face_component_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    return trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))


def submesh_from_face_mask(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    face_index = np.nonzero(face_mask)[0]
    if len(face_index) == 0:
        raise ValueError("Face mask removed the entire mesh")
    sub = mesh.submesh([face_index], append=True, repair=False)
    if isinstance(sub, list):
        sub = trimesh.util.concatenate(sub)
    return trimesh.Trimesh(vertices=sub.vertices.copy(), faces=sub.faces.copy(), process=False)


def prune_small_components(mesh: trimesh.Trimesh, min_faces: int) -> tuple[trimesh.Trimesh, dict[str, int] | None]:
    if min_faces <= 0:
        return mesh, None
    labels = face_component_labels(mesh)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    keep_components = counts >= min_faces
    keep_mask = keep_components[labels]
    pruned = submesh_from_face_mask(mesh, keep_mask)
    return pruned, {
        "min_faces": int(min_faces),
        "source_components": int(len(counts)),
        "kept_components": int(keep_components.sum()),
        "removed_components": int((~keep_components).sum()),
        "kept_faces": int(counts[keep_components].sum()),
        "removed_faces": int(counts[~keep_components].sum()),
    }


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    labels = face_component_labels(mesh)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    keep_mask = labels == int(np.argmax(counts))
    return submesh_from_face_mask(mesh, keep_mask)


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


def sample_points_with_normals(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_index = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_index]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return points.astype(np.float64), normals.astype(np.float64)


def run_poisson_open3d(
    source: trimesh.Trimesh,
    depth: int,
    sample_count: int,
    normal_knn: int,
    density_quantile: float,
    scale: float,
    linear_fit: bool,
    crop_padding_frac: float,
) -> tuple[trimesh.Trimesh, float]:
    if o3d is None:
        raise RuntimeError("open3d is not installed")
    points, normals = sample_points_with_normals(source, sample_count)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(normal_knn)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit,
    )

    densities = np.asarray(densities)
    threshold = float(np.quantile(densities, density_quantile)) if density_quantile > 0 else float(densities.min())
    if density_quantile > 0:
        mesh_o3d.remove_vertices_by_mask(densities < threshold)

    pad = float(np.max(source.extents) * crop_padding_frac)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(source.bounds[0] - pad).astype(np.float64),
        max_bound=(source.bounds[1] + pad).astype(np.float64),
    )
    mesh_o3d = mesh_o3d.crop(bbox)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()

    repaired = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=False,
    )
    return repaired, threshold


def run_poisson_pymeshlab(
    source_obj: Path,
    raw_obj: Path,
    depth: int,
    scale: float,
    samples_per_node: float,
    point_weight: float,
    poisson_iters: int,
    threads: int,
) -> tuple[trimesh.Trimesh, float | None]:
    if ml is None:
        raise RuntimeError("pymeshlab is not installed")

    ms = ml.MeshSet()
    ms.load_new_mesh(str(source_obj))
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.compute_normal_per_vertex()
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth,
        scale=scale,
        pointweight=point_weight,
        samplespernode=samples_per_node,
        iters=poisson_iters,
        confidence=False,
        preclean=True,
        threads=threads,
    )
    ms.save_current_mesh(str(raw_obj))
    repaired = load_mesh(raw_obj)
    return repaired, None


def prefill_holes_pymeshlab(
    source_obj: Path,
    filled_obj: Path,
    maxholesize: int,
    selfintersection: bool,
) -> trimesh.Trimesh:
    if ml is None:
        raise RuntimeError("pymeshlab is not installed")
    ms = ml.MeshSet()
    ms.load_new_mesh(str(source_obj))
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    ms.meshing_close_holes(
        maxholesize=maxholesize,
        selected=False,
        newfaceselected=True,
        selfintersection=selfintersection,
        refinehole=False,
    )
    ms.save_current_mesh(str(filled_obj))
    return load_mesh(filled_obj)


def project_back(
    repaired: trimesh.Trimesh,
    source: trimesh.Trimesh,
    blend: float,
    max_distance: float,
) -> tuple[trimesh.Trimesh, dict[str, float]]:
    if blend <= 0:
        return repaired.copy(), {"projected_vertices": 0, "max_distance": max_distance}

    prox = trimesh.proximity.ProximityQuery(source)
    closest, dist, _ = prox.on_surface(repaired.vertices)
    if max_distance > 0:
        mask = dist <= max_distance
    else:
        mask = np.ones(len(dist), dtype=bool)

    new_vertices = repaired.vertices.copy()
    new_vertices[mask] = (
        (1.0 - blend) * new_vertices[mask]
        + blend * closest[mask]
    )
    projected = trimesh.Trimesh(vertices=new_vertices, faces=repaired.faces.copy(), process=False)
    return projected, {
        "projected_vertices": int(mask.sum()),
        "total_vertices": int(len(mask)),
        "mean_projection_distance": float(dist[mask].mean()) if mask.any() else 0.0,
        "p95_projection_distance": float(np.quantile(dist[mask], 0.95)) if mask.any() else 0.0,
        "max_projection_distance": float(dist[mask].max()) if mask.any() else 0.0,
        "max_distance": float(max_distance),
        "blend": float(blend),
    }


def main() -> int:
    args = parse_args()
    input_mesh = args.input_mesh.resolve()
    if not input_mesh.exists():
        raise SystemExit(f"Input mesh not found: {input_mesh}")

    if args.output_prefix is None:
        stem = input_mesh.with_suffix("")
        output_prefix = stem.parent / f"{stem.name}_poisson_d{args.depth}"
    else:
        output_prefix = args.output_prefix.resolve()

    source = load_mesh(input_mesh)
    raw_obj = output_prefix.with_name(f"{output_prefix.name}_raw.obj")
    raw_glb = output_prefix.with_name(f"{output_prefix.name}_raw.glb")
    proj_obj = output_prefix.with_name(f"{output_prefix.name}_projected.obj")
    proj_glb = output_prefix.with_name(f"{output_prefix.name}_projected.glb")
    raw_main_glb = output_prefix.with_name(f"{output_prefix.name}_main_raw.glb")
    proj_main_glb = output_prefix.with_name(f"{output_prefix.name}_main_projected.glb")
    source_obj = output_prefix.with_name(f"{output_prefix.name}_source.obj")
    pruned_obj = output_prefix.with_name(f"{output_prefix.name}_pruned.obj")
    prefill_obj = output_prefix.with_name(f"{output_prefix.name}_prefill.obj")
    report_json = output_prefix.with_suffix(".json")

    source_for_poisson = source
    prune_stats = None
    if args.prune_min_faces > 0:
        source_for_poisson, prune_stats = prune_small_components(source_for_poisson, args.prune_min_faces)
        source_for_poisson.export(pruned_obj)
    source.export(source_obj)
    prefill_stats = None
    if args.prefill_holes_max_size > 0:
        source_for_poisson.export(source_obj)
        source_for_poisson = prefill_holes_pymeshlab(
            source_obj=source_obj,
            filled_obj=prefill_obj,
            maxholesize=args.prefill_holes_max_size,
            selfintersection=args.prefill_selfintersection,
        )
        prefill_stats = mesh_stats(source_for_poisson)
        source_for_poisson.export(source_obj)
    elif args.prune_min_faces > 0:
        source_for_poisson.export(source_obj)

    if args.backend == "pymeshlab":
        raw_mesh, density_threshold = run_poisson_pymeshlab(
            source_obj=source_obj,
            raw_obj=raw_obj,
            depth=args.depth,
            scale=args.scale,
            samples_per_node=args.samples_per_node,
            point_weight=args.point_weight,
            poisson_iters=args.poisson_iters,
            threads=args.threads,
        )
    else:
        raw_mesh, density_threshold = run_poisson_open3d(
            source=source_for_poisson,
            depth=args.depth,
            sample_count=args.sample_count,
            normal_knn=args.normal_knn,
            density_quantile=args.density_quantile,
            scale=args.scale,
            linear_fit=args.linear_fit,
            crop_padding_frac=args.crop_padding_frac,
        )
        raw_mesh.export(raw_obj)

    projected_mesh, projection_info = project_back(
        repaired=raw_mesh,
        source=source,
        blend=args.project_back_blend,
        max_distance=args.project_back_max_distance,
    )

    raw_mesh.export(raw_glb)
    projected_mesh.export(proj_obj)
    projected_mesh.export(proj_glb)

    raw_main_stats = None
    projected_main_stats = None
    raw_main_distance = None
    projected_main_distance = None
    if args.keep_largest_component:
        raw_main = largest_component(raw_mesh)
        projected_main = largest_component(projected_mesh)
        raw_main.export(raw_main_glb)
        projected_main.export(proj_main_glb)
        raw_main_stats = mesh_stats(raw_main)
        projected_main_stats = mesh_stats(projected_main)
        raw_main_distance = approximate_surface_distances(source, raw_main, args.distance_samples)
        projected_main_distance = approximate_surface_distances(source, projected_main, args.distance_samples)

    report = {
        "input_mesh": str(input_mesh),
        "backend": args.backend,
        "prune_min_faces": args.prune_min_faces,
        "depth": args.depth,
        "sample_count": args.sample_count,
        "normal_knn": args.normal_knn,
        "density_quantile": args.density_quantile,
        "density_threshold": density_threshold,
        "scale": args.scale,
        "linear_fit": args.linear_fit,
        "samples_per_node": args.samples_per_node,
        "point_weight": args.point_weight,
        "poisson_iters": args.poisson_iters,
        "threads": args.threads,
        "prefill_holes_max_size": args.prefill_holes_max_size,
        "prefill_selfintersection": args.prefill_selfintersection,
        "crop_padding_frac": args.crop_padding_frac,
        "projection": projection_info,
        "source": mesh_stats(source),
        "pruned": prune_stats,
        "prefill": prefill_stats,
        "raw": mesh_stats(raw_mesh),
        "projected": mesh_stats(projected_mesh),
        "raw_main": raw_main_stats,
        "projected_main": projected_main_stats,
        "raw_distance": approximate_surface_distances(source, raw_mesh, args.distance_samples),
        "projected_distance": approximate_surface_distances(source, projected_mesh, args.distance_samples),
        "raw_main_distance": raw_main_distance,
        "projected_main_distance": projected_main_distance,
        "outputs": {
            "source_obj": str(source_obj),
            "pruned_obj": str(pruned_obj) if prune_stats is not None else None,
            "raw_obj": str(raw_obj),
            "raw_glb": str(raw_glb),
            "projected_obj": str(proj_obj),
            "projected_glb": str(proj_glb),
            "raw_main_glb": str(raw_main_glb) if raw_main_stats is not None else None,
            "projected_main_glb": str(proj_main_glb) if projected_main_stats is not None else None,
        },
    }
    report_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
