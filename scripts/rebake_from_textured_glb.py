#!/usr/bin/env python3
"""Rebake textures from a textured GLB onto a replacement geometry GLB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image
from trimesh.visual.color import uv_to_interpolated_color

import mtldiffrast.torch as dr
from cumesh import CuMesh, cuBVH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Textured source GLB")
    parser.add_argument("target", type=Path, help="Target geometry GLB")
    parser.add_argument("output", type=Path, help="Output textured GLB")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--texture-size", type=int, default=1024, help="Baked texture resolution")
    parser.add_argument("--force-opaque", action="store_true", help="Force alpha to 255 / OPAQUE")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a Trimesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"{path} is empty")
    return mesh


def require_textured_source(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        raise TypeError("Source mesh must use TextureVisuals")
    if mesh.visual.uv is None:
        raise ValueError("Source mesh has no UVs")
    material = mesh.visual.material
    if material is None or material.baseColorTexture is None:
        raise ValueError("Source mesh has no baseColorTexture")
    if material.metallicRoughnessTexture is None:
        raise ValueError("Source mesh has no metallicRoughnessTexture")


def build_bake_uvs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    mesh = CuMesh()
    mesh.init(vertices, faces)
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": np.radians(90.0),
            "refine_iterations": 0,
            "global_iterations": 1,
            "smooth_strength": 1,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    ctx = dr.MtlRasterizeContext()
    uvs_rast = torch.cat(
        [
            out_uvs * 2 - 1,
            torch.zeros_like(out_uvs[:, :1]),
            torch.ones_like(out_uvs[:, :1]),
        ],
        dim=-1,
    ).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), dtype=torch.float32)
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx,
            uvs_rast,
            out_faces[i : i + 100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)

    mask = (rast[0, ..., 3] > 0).cpu().numpy()
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    return out_vertices, out_faces, out_uvs, out_normals, mask, pos


def bake_textures(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    texture_size: int,
    force_opaque: bool,
    verbose: bool,
) -> tuple[trimesh.Trimesh, dict]:
    source_vertices = torch.from_numpy(np.asarray(source_mesh.vertices, dtype=np.float32))
    source_faces = torch.from_numpy(np.asarray(source_mesh.faces, dtype=np.int64))
    source_uvs = torch.from_numpy(np.asarray(source_mesh.visual.uv, dtype=np.float32))
    target_vertices = torch.from_numpy(np.asarray(target_mesh.vertices, dtype=np.float32))
    target_faces = torch.from_numpy(np.asarray(target_mesh.faces, dtype=np.int64))

    if verbose:
        print("Unwrapping target mesh...")
    out_vertices, out_faces, out_uvs, out_normals, mask, pos = build_bake_uvs(
        target_vertices,
        target_faces,
        texture_size,
        verbose,
    )
    valid_pos = pos[mask]

    if verbose:
        print("Projecting texels to source mesh...")
    bvh = cuBVH(source_vertices, source_faces)
    distances, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    face_id = face_id.long()
    tri_uvs = source_uvs[source_faces[face_id]]
    source_tex_uv = (tri_uvs * uvw.unsqueeze(-1)).sum(dim=1).cpu().numpy()

    material = source_mesh.visual.material
    base_rgba = uv_to_interpolated_color(source_tex_uv, material.baseColorTexture)
    mr_rgba = uv_to_interpolated_color(source_tex_uv, material.metallicRoughnessTexture)

    base_color = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    alpha = np.zeros((texture_size, texture_size, 1), dtype=np.uint8)
    metallic = np.zeros((texture_size, texture_size, 1), dtype=np.uint8)
    roughness = np.zeros((texture_size, texture_size, 1), dtype=np.uint8)

    base_color[mask] = base_rgba[:, :3]
    alpha[mask, 0] = 255 if force_opaque else base_rgba[:, 3]
    roughness[mask, 0] = mr_rgba[:, 1]
    metallic[mask, 0] = mr_rgba[:, 2]

    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic[..., 0], mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness[..., 0], mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    if force_opaque:
        alpha[...] = 255
        alpha_mode = "OPAQUE"
    else:
        alpha = cv2.inpaint(alpha[..., 0], mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha_valid = alpha[mask]
        alpha_mode = "BLEND" if alpha_valid.size > 0 and int(alpha_valid.min()) < 250 else "OPAQUE"

    baked_material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array(getattr(material, "baseColorFactor", [255, 255, 255, 255]), dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=float(getattr(material, "metallicFactor", 1.0)),
        roughnessFactor=float(getattr(material, "roughnessFactor", 1.0)),
        alphaMode=alpha_mode,
        doubleSided=bool(getattr(material, "doubleSided", True)),
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    uvs_np[:, 1] = 1.0 - uvs_np[:, 1]

    baked_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=baked_material),
    )

    alpha_valid = alpha[mask]
    report = {
        "source": str(source_mesh.metadata.get("file_path", "")),
        "target_vertex_count": int(len(target_mesh.vertices)),
        "target_face_count": int(len(target_mesh.faces)),
        "output_vertex_count": int(len(vertices_np)),
        "output_face_count": int(len(faces_np)),
        "texture_size": int(texture_size),
        "valid_texel_count": int(mask.sum()),
        "coverage_fraction": float(mask.mean()),
        "projection_distance_mean": float(distances.mean().item()) if distances.numel() else 0.0,
        "projection_distance_p95": float(torch.quantile(distances, 0.95).item()) if distances.numel() else 0.0,
        "projection_distance_max": float(distances.max().item()) if distances.numel() else 0.0,
        "alpha_mode": alpha_mode,
        "opaque_fraction": float((alpha_valid >= 250).mean()) if alpha_valid.size else 1.0,
        "alpha_min": int(alpha_valid.min()) if alpha_valid.size else 255,
        "alpha_max": int(alpha_valid.max()) if alpha_valid.size else 255,
    }
    return baked_mesh, report


def main() -> None:
    args = parse_args()

    source_mesh = load_mesh(args.source)
    source_mesh.metadata["file_path"] = str(args.source)
    require_textured_source(source_mesh)
    target_mesh = load_mesh(args.target)

    baked_mesh, report = bake_textures(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        texture_size=args.texture_size,
        force_opaque=args.force_opaque,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    baked_mesh.export(args.output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.verbose:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
