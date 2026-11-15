# Running the experiments on a GPU (Kaggle or Colab)

None of this has been executed on a GPU yet. Treat the first run as a test of the gsplat and model paths, and send back the full error text if a step fails.

You need an NVIDIA GPU with about 16 GB of memory. A Kaggle notebook with "GPU T4 x2" or "P100" and **Internet turned on** is enough for LLFF.

## 0. Setup (one cell)

```bash
!git clone https://github.com/KunalChavan245/3D-scene-reconstruction-from-few-images.git
%cd 3D-scene-reconstruction-from-few-images
!pip install -q -e .
!pip install -q lpips opencv-python-headless transformers einops safetensors huggingface_hub
!pip install -q --no-deps git+https://github.com/facebookresearch/vggt
!python -c "import torch, gsplat; print(torch.__version__, torch.cuda.is_available(), gsplat.__version__)"
```

`--no-deps` on VGGT stops it from downgrading NumPy. The first gsplat render may spend several minutes compiling its CUDA code; that happens once per session.

The VGGT checkpoint on Hugging Face may ask you to accept its terms. If the download is refused, open https://huggingface.co/facebook/VGGT-1B while signed in, accept, then run `from huggingface_hub import login; login()` with a read token.

## 1. Check the install (2 minutes)

```bash
!python -m pytest -q -k "not experiment and not scripts"
```

## 2. Get the data

The scripts expect `data/<dataset>/<scene>/` with `images/`, optionally `images_8/`, and `sparse/0/`.

**LLFF.** Download `nerf_llff_data.zip` from the official NeRF data folder linked in the NeRF repository (https://github.com/bmild/nerf, "nerf_llff_data"), or attach a Kaggle dataset that mirrors it. Then:

```bash
!mkdir -p data && unzip -q nerf_llff_data.zip -d data && mv data/nerf_llff_data data/llff
!ls data/llff/fern      # expect: images  images_4  images_8  sparse  poses_bounds.npy ...
```

**Mip-NeRF 360.**

```bash
!wget -q http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
!mkdir -p data/mipnerf360 && unzip -q 360_v2.zip -d data/mipnerf360
```

## 3. Milestone A: validate the GPU path on one scene

Train with **all** views first. If this number is far from published 3DGS results on LLFF `fern` at 1/8 resolution, stop and report it before running anything else.

```bash
!python scripts/make_splits.py --dataset llff --views 0 3 6 9
!python scripts/train.py --dataset llff --scenes fern --n-views 0 --init sfm_all --max-steps 30000 --refine-stop-frac 0.5
```

## 4. Milestone B: the few-view failure (vanilla 3DGS, leaky protocol)

```bash
!python scripts/run_sweep.py --dataset llff --views 3 6 9 --seeds 0 1 2 --cells colmap:none
!python scripts/aggregate.py
```

## 5. Milestone C: priors and the full 2 x 3 grid

```bash
!python scripts/estimate_vggt.py         --dataset llff --views 3 6 9
!python scripts/precompute_mono_depth.py --dataset llff --views 3 6 9
!python scripts/run_sweep.py --dataset llff --views 3 --seeds 0 1 2
!python scripts/aggregate.py
```

`estimate_vggt.py` prints, for each scene, how far the honest poses are from full-view COLMAP (rotation error in degrees, camera-centre error relative to the camera spread). Keep that output.

The sweep skips runs that already have a `results.json`, so after a session timeout just run the same command again. To keep outputs across Kaggle sessions, pass `--work-root /kaggle/working/run1` to every script and save the notebook version.

## 6. Ablations (one scene first)

```bash
# depth-loss weight
!for L in 0.001 0.01 0.05 0.1 0.5; do python scripts/train.py --dataset llff --scenes fern --n-views 3 \
    --pose-source vggt --depth-prior vggt --lambda-depth $L --tag lam$L; done
# confidence weighting off / hard threshold / soft
!for M in none threshold soft; do python scripts/train.py --dataset llff --scenes fern --n-views 3 \
    --pose-source vggt --depth-prior vggt --conf-mode $M --tag conf-$M; done
# refine the estimated training poses during optimisation
!python scripts/train.py --dataset llff --scenes fern --n-views 3 --pose-source vggt --depth-prior vggt \
    --optimize-train-poses --tag poseopt
```

Runs with a `--tag` are kept out of the main tables.

## 7. What to send back

1. `results/RESULTS.md` and `results/results.csv`
2. The printed output of `estimate_vggt.py`
3. A few images from `outputs/llff/<scene>/<run>/renders/`
4. Any error, as full text

These go into `docs/RESEARCH_LOG.md` and then into the README and the paper. No number is written anywhere until it comes from one of these files.
