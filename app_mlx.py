"""
Gradio app for Trellis2 with MLX backend (macOS).
Simplified from upstream app.py — no CUDA render preview, direct GLB export.
"""
import os
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import time
from datetime import datetime
import shutil
import gc
import numpy as np
from PIL import Image
import mlx.core as mx
import torch
import o_voxel

MAX_SEED = np.iinfo(np.int32).max
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp')


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def free_memory() -> None:
    gc.collect()
    try:
        mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    skip_background_removal: bool,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    shape_slat_guidance_strength: float,
    shape_slat_sampling_steps: int,
    tex_slat_guidance_strength: float,
    tex_slat_sampling_steps: int,
    target_faces: int,
    texture_size: int,
    hole_fill_max_perimeter: float,
    close_shell: bool,
    close_shell_resolution: int,
    close_shell_iters: int,
    close_shell_project_back: float,
    force_opaque: bool,
    remesh: bool,
    remesh_band: float,
    remesh_project: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution]
    meshes = None
    mesh = None
    glb = None
    try:
        t0 = time.time()
        meshes = pipeline.run(
            image,
            seed=seed,
            skip_background_removal=skip_background_removal,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
            },
            pipeline_type=pipeline_type,
        )
        dt_gen = time.time() - t0
        mesh = meshes[0]

        t0 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=target_faces,
            texture_size=texture_size,
            hole_fill_max_perimeter=hole_fill_max_perimeter,
            close_shell=close_shell,
            close_shell_resolution=close_shell_resolution,
            close_shell_iters=close_shell_iters,
            close_shell_project_back=close_shell_project_back,
            force_opaque=force_opaque,
            remesh=remesh,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            verbose=True,
        )
        dt_post = time.time() - t0

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
        glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
        glb.export(glb_path)

        export_mode = "Remesh" if remesh else "Standard"
        info = (f"Generation: {dt_gen:.0f}s | Post-processing: {dt_post:.0f}s | {export_mode} | "
                f"Raw: {mesh.vertices.shape[0]:,} verts / {mesh.faces.shape[0]:,} faces | "
                f"Exported: {len(glb.vertices):,} verts / {len(glb.faces):,} faces")
        return glb_path, glb_path, info
    finally:
        del glb, mesh, meshes
        free_memory()


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


with gr.Blocks(title="Trellis2 MLX") as demo:
    gr.Markdown("""
    ## Trellis2 (MLX Backend)
    Upload an image and click Generate to create a 3D asset.
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)

            resolution = gr.Radio(["512", "1024"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
            target_faces = gr.Slider(5000, 1000000, label="Target Faces", value=250000, step=5000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion(label="Advanced Settings", open=False):
                skip_background_removal = gr.Checkbox(
                    label="Skip Background Removal",
                    value=False,
                    info="Use this for images that are already cut out. RGBA images with real transparency already skip rembg automatically.",
                )
                gr.Markdown("### Stage 1: Sparse Structure")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("### Stage 2: Shape")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance", value=7.5, step=0.1)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("### Stage 3: Texture")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(0.1, 10.0, label="Guidance", value=1.0, step=0.1)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("### Export")
                hole_fill_max_perimeter = gr.Slider(
                    0.0, 0.10, label="Hole Fill Perimeter", value=0.03, step=0.005,
                    info="Patch small topology holes before UV baking. Larger values are more aggressive.",
                )
                close_shell = gr.Checkbox(
                    label="Close Fragmented Shell",
                    value=False,
                    info="Merge nearby disconnected shell fragments into one closed surface before baking.",
                )
                with gr.Row():
                    close_shell_resolution = gr.Slider(96, 256, label="Shell Resolution", value=192, step=32)
                    close_shell_iters = gr.Slider(0, 3, label="Shell Closing Iterations", value=1, step=1)
                with gr.Row():
                    close_shell_project_back = gr.Slider(0.0, 1.0, label="Project Back", value=1.0, step=0.05)
                    force_opaque = gr.Checkbox(
                        label="Force Opaque",
                        value=False,
                        info="Override baked alpha and export an opaque material.",
                    )
                remesh = gr.Checkbox(
                    label="Topology Remesh",
                    value=False,
                    info="Rebuild the mesh surface before baking textures. Slower, but can reduce open seams and holes.",
                )
                with gr.Row():
                    remesh_band = gr.Slider(0.25, 2.0, label="Remesh Band", value=1.0, step=0.05)
                    remesh_project = gr.Slider(0.0, 1.0, label="Project Back", value=0.9, step=0.05)

        with gr.Column(scale=2):
            glb_output = gr.Model3D(label="Generated 3D Model", height=700, clear_color=(0.25, 0.25, 0.25, 1.0))
            download_btn = gr.DownloadButton(label="Download GLB")
            info_text = gr.Textbox(label="Info", interactive=False)

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            skip_background_removal,
            ss_guidance_strength, ss_sampling_steps,
            shape_slat_guidance_strength, shape_slat_sampling_steps,
            tex_slat_guidance_strength, tex_slat_sampling_steps,
            target_faces, texture_size, hole_fill_max_perimeter,
            close_shell, close_shell_resolution, close_shell_iters,
            close_shell_project_back, force_opaque,
            remesh, remesh_band, remesh_project,
        ],
        outputs=[glb_output, download_btn, info_text],
    )


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    from mlx_backend.pipeline import create_mlx_pipeline
    default_weights = os.path.join(PROJECT_ROOT, "weights", "TRELLIS.2-4B")
    pipeline = create_mlx_pipeline(weights_path=os.environ.get("TRELLIS2_WEIGHTS", default_weights))

    demo.launch()
