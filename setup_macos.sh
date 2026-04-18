#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-macos}"
DOWNLOAD_WEIGHTS=1

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON_BIN="python3.10"
    else
        PYTHON_BIN="python3"
    fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --skip-weights)
            DOWNLOAD_WEIGHTS=0
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./setup_macos.sh [OPTIONS]

Options:
  --python PATH      Python interpreter to use (default: python3)
  --venv PATH        Virtualenv location (default: ./.venv-macos)
  --skip-weights     Install dependencies only
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup_macos.sh is only intended for macOS." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Apple Silicon (arm64) is required for the MLX + Metal path." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Build-time dependencies for Metal packages must exist in the active venv first.
python -m pip install \
    torch \
    torchvision \
    transformers \
    safetensors \
    huggingface_hub \
    pillow \
    numpy \
    easydict \
    trimesh \
    xatlas \
    fast-simplification \
    pygltflib \
    plyfile \
    zstandard \
    opencv-python-headless \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "gradio>=6,<7" \
    mlx \
    tqdm \
    imageio \
    imageio-ffmpeg \
    scipy \
    einops

python -m pip install --no-build-isolation \
    mtldiffrast@https://github.com/pedronaugusto/mtldiffrast/archive/main.tar.gz \
    mtlbvh@https://github.com/pedronaugusto/mtlbvh/archive/main.tar.gz \
    cumesh@https://github.com/pedronaugusto/mtlmesh/archive/main.tar.gz \
    flex_gemm@https://github.com/pedronaugusto/mtlgemm/archive/main.tar.gz

if command -v git >/dev/null 2>&1 && [[ -e "$ROOT_DIR/.git" ]]; then
    git -C "$ROOT_DIR" submodule update --init --recursive
fi

BUILD_TARGET=cpu python -m pip install --no-build-isolation -e "$ROOT_DIR/o-voxel"

if [[ "$DOWNLOAD_WEIGHTS" -eq 1 ]]; then
    python "$ROOT_DIR/scripts/download_weights.py" --output-dir "$ROOT_DIR/weights/TRELLIS.2-4B"
fi

cat <<EOF
macOS setup complete.

Activate the environment:
  source "$VENV_DIR/bin/activate"

Run the MLX app:
  python "$ROOT_DIR/app_mlx.py"

Run the texturing app:
  python "$ROOT_DIR/app_texturing.py"
EOF
