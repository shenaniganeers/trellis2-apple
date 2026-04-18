import os
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import trimesh
import torch
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline


def get_best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 1. Load Pipeline
pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.to(get_best_device())

# 2. Load Mesh, image & Run
mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
image = Image.open("assets/example_texturing/image.webp")
output = pipeline.run(mesh, image)

# 3. Render Mesh
output.export("textured.glb", extension_webp=True)
