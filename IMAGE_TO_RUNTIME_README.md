# Image to Runtime Mesh Workflow

This document captures the workflow we actually used in this repo to go from a single source image to a usable local runtime mesh on Apple Silicon.

It is not an upstream TRELLIS2 guide. It is the Apple/MLX repair pipeline we arrived at after working through real assets in this checkout.

## Goal

For hard-surface assets, the practical target is:

- one good source image
- high-quality TRELLIS2 image-to-3D baseline
- repaired closed high-poly shell
- decimated runtime mesh at about `20k` triangles
- rebaked textured GLB on the repaired low-poly mesh

For `CrushedIron`, `20k` was the best tradeoff. `5k` was usable as an experiment but visibly distorted.

## Environment

Use the Apple setup and the project venv:

```sh
./setup_macos.sh
source .venv-macos/bin/activate
```

For repeated heavy runs, prefer the one-shot runner instead of the long-lived Gradio app:

- use [`scripts/run_once_mlx.py`](scripts/run_once_mlx.py)
- avoid keeping [`app_mlx.py`](app_mlx.py) open for batch experiments

Reason: the Gradio process is long-lived, keeps the pipeline in memory, and was not the safest way to iterate on large `1024` jobs.

## Output Naming Convention

The pipeline naturally breaks into a few stages. These names are the ones we ended up using:

- `*_baseline.glb`
  - direct high-quality TRELLIS2 output
- `*_prefill30.glb`
  - source mesh after microhole cleanup
- `*_poisson_*_projected.glb`
  - closed high-poly shell from Screened Poisson + projection-back
- `*_poisson_deblob*.glb`
  - locally flattened/fairing-cleaned version of the repaired shell
- `*_plane_patch_*.glb`
  - hard-surface plane-guided local patch repair, when Poisson still leaves a bad region
- `*_runtime_20k_geometry.glb`
  - decimated watertight runtime geometry
- `*_runtime_20k.glb`
  - rebaked textured runtime mesh

## The Working Pipeline

### 1. Prepare the source image

Best case:

- the input is already a clean RGBA cutout
- background removal can be skipped

If the source is a white-background product render, use the helper cutout script:

```sh
python scripts/cutout_white_bg.py input.png tmp/asset_cutout.png --trim
```

Relevant script:

- [`scripts/cutout_white_bg.py`](scripts/cutout_white_bg.py)

### 2. Generate a high-quality baseline mesh

Generate the baseline with the one-shot MLX runner and keep the export target high. At this stage we want geometry fidelity, not runtime triangle count.

Typical command:

```sh
python scripts/run_once_mlx.py \
  input.png \
  --output tmp/Asset_baseline.glb \
  --resolution 1024 \
  --ss-steps 20 \
  --shape-steps 20 \
  --tex-steps 12 \
  --target-faces 1000000 \
  --texture-size 2048 \
  --skip-background-removal
```

Relevant script:

- [`scripts/run_once_mlx.py`](scripts/run_once_mlx.py)

Notes:

- `1024`, `20/20/12` was the best high-quality baseline we used for these assets.
- `target-faces` stays high here because low-poly conversion happens later.
- If the input is already RGBA, use `--skip-background-removal`.

### 3. Diagnose the baseline before repairing it

There were two recurring failure modes:

- large gaps between disconnected components
- porous or speckled surfaces with many tiny missing triangles

Useful diagnostics:

```sh
python scripts/analyze_holes.py tmp/Asset_baseline.glb --top 25 --threshold 0.05

python scripts/color_components.py \
  tmp/Asset_baseline.glb \
  --output tmp/Asset_components.glb \
  --top 120
```

Relevant scripts:

- [`scripts/analyze_holes.py`](scripts/analyze_holes.py)
- [`scripts/color_components.py`](scripts/color_components.py)

Important lesson from the debugging work:

- not every visible "hole" is a boundary hole
- some of the worst defects are gaps between large disconnected shells
- hole filling alone does not solve those

### 4. Clean small holes in the source and build a closed shell

The best general-purpose repair path we found was:

1. prefill small source holes
2. run Screened Poisson
3. keep the largest reconstructed component
4. project the reconstruction back toward the original surface

Typical command:

```sh
python scripts/poisson_compare.py \
  tmp/Asset_baseline.glb \
  --backend pymeshlab \
  --depth 8 \
  --scale 1.10 \
  --point-weight 4 \
  --threads 1 \
  --prefill-holes-max-size 30 \
  --keep-largest-component \
  --output-prefix tmp/Asset_poisson
```

Relevant script:

- [`scripts/poisson_compare.py`](scripts/poisson_compare.py)

What this step does well:

- closes large inter-component gaps better than simple hole filling
- produces a single closed shell candidate

What it does not do well:

- it remeshes the whole surface
- it can create outward blobs or bulges

### 5. Suppress Poisson blobs locally

After Poisson, the next step is local fairing only on unsupported regions that drift too far from the source surface.

Typical command:

```sh
python scripts/blob_suppress.py \
  tmp/Asset_prefill30.glb \
  tmp/Asset_poisson_main_projected.glb \
  --output tmp/Asset_poisson_deblob.glb \
  --report tmp/Asset_poisson_deblob.json \
  --debug-mask-output tmp/Asset_poisson_deblob_mask.glb \
  --selection-mode allfar
```

Relevant script:

- [`scripts/blob_suppress.py`](scripts/blob_suppress.py)

Important lesson:

- the default "outward convex only" mask was too narrow for some assets
- `--selection-mode allfar` was necessary to catch the biggest unsupported patches

### 6. If needed, use a plane-guided hard-surface patch repair

Some Poisson defects are not really blobs. They are missing hard-surface patches where the intended answer is "flatten this region onto the surrounding plane."

For those cases:

1. inspect the patch and fitted support planes
2. run a plane-guided local repair

Plane debug:

```sh
python scripts/plane_patch_debug.py \
  tmp/Asset_prefill30.glb \
  tmp/Asset_poisson_deblob.glb \
  --output tmp/Asset_plane_debug.glb \
  --report tmp/Asset_plane_debug.json
```

Plane repair:

```sh
python scripts/plane_patch_repair.py \
  tmp/Asset_prefill30.glb \
  tmp/Asset_poisson_deblob.glb \
  --output tmp/Asset_plane_patch.glb \
  --report tmp/Asset_plane_patch.json \
  --debug-mask-output tmp/Asset_plane_patch_mask.glb
```

Relevant scripts:

- [`scripts/plane_patch_debug.py`](scripts/plane_patch_debug.py)
- [`scripts/plane_patch_repair.py`](scripts/plane_patch_repair.py)

When to use this step:

- Poisson closed the shell
- local deblob helped but left a rough hard-surface patch
- the surrounding geometry is clearly planar or almost planar

This step was necessary for `CrushedIron`. It was not necessary for `ScrapBolt`.

### 7. Decimate to the runtime target

Once the repaired high-poly geometry looks good, decimate it to the shipping target before rebaking textures.

Typical command:

```sh
python scripts/decimate_geometry_glb.py \
  tmp/Asset_repaired.glb \
  tmp/Asset_runtime_20k_geometry.glb \
  --target-faces 20000 \
  --hole-fill-perimeter 0.15 \
  --report tmp/Asset_runtime_20k_geometry.json \
  --verbose
```

Relevant script:

- [`scripts/decimate_geometry_glb.py`](scripts/decimate_geometry_glb.py)

Important lesson:

- decimation can reopen small holes even when the repaired high-poly shell is closed
- fixing those holes at the decimated stage worked well for the `20k` target
- for `CrushedIron`, the decimated `20k` mesh closed cleanly with only a tiny face-count increase

### 8. Rebake textures onto the runtime mesh

Use the textured baseline GLB as the source material and rebake onto the repaired low-poly geometry.

Typical command:

```sh
python scripts/rebake_from_textured_glb.py \
  tmp/Asset_baseline.glb \
  tmp/Asset_runtime_20k_geometry.glb \
  tmp/Asset_runtime_20k.glb \
  --texture-size 1024 \
  --force-opaque \
  --report tmp/Asset_runtime_20k.json \
  --verbose
```

Relevant script:

- [`scripts/rebake_from_textured_glb.py`](scripts/rebake_from_textured_glb.py)

Important lesson:

- evaluate watertightness on the geometry-only GLB
- do not use `trimesh` body count from the textured GLB as the source of truth
- UV seam duplication in the textured export will make the mesh look numerically fragmented even when the underlying geometry is fine

## Decision Tree

Use this as the default decision path:

1. Generate a high-quality baseline.
2. If the mesh is already acceptable, skip straight to decimation and rebake.
3. If the source has many tiny perforations, run source prefill and Poisson.
4. If Poisson creates outward blobs, run local deblob.
5. If one hard-surface region still looks cratered or stepped, run the plane patch repair.
6. Decimate to `20k`.
7. Fill any decimation-introduced holes.
8. Rebake textures.

## What Did Not Work Well

These were explored and were not the preferred path:

- voxel close / close-shell export
  - robust topology, poor surface quality
  - too boxy or grainy for hard-surface assets
- naive local component bridging
  - preserved original geometry but did not solve the real failures well enough
- aggressive `5k` runtime target
  - useful as an LOD experiment
  - too distorted for the current quality bar

## The Current Default Recommendation

For a new hard-surface asset, start here:

1. `cutout_white_bg.py` only if the source is not already RGBA
2. `run_once_mlx.py` with `1024`, `20/20/12`
3. `poisson_compare.py --prefill-holes-max-size 30 --keep-largest-component`
4. `blob_suppress.py --selection-mode allfar`
5. `plane_patch_repair.py` only if Poisson still leaves one obvious hard-surface defect
6. `decimate_geometry_glb.py --target-faces 20000 --hole-fill-perimeter 0.15`
7. `rebake_from_textured_glb.py --texture-size 1024 --force-opaque`

## Example Final Outputs

Current final `CrushedIron` runtime candidate:

- geometry: [`tmp/CrushedIron_runtime_20k_geometry.glb`](tmp/CrushedIron_runtime_20k_geometry.glb)
- textured: [`tmp/CrushedIron_runtime_20k.glb`](tmp/CrushedIron_runtime_20k.glb)
- report: [`tmp/CrushedIron_runtime_20k.json`](tmp/CrushedIron_runtime_20k.json)

Second asset that followed the same general pattern:

- geometry: [`tmp/ScrapBolt_runtime_20k_geometry_main.glb`](tmp/ScrapBolt_runtime_20k_geometry_main.glb)
- textured: [`tmp/ScrapBolt_runtime_20k.glb`](tmp/ScrapBolt_runtime_20k.glb)
- report: [`tmp/ScrapBolt_runtime_20k.json`](tmp/ScrapBolt_runtime_20k.json)

The second asset did not need the plane patch stage, which suggests the general pattern is reusable even if not every asset needs every step.
