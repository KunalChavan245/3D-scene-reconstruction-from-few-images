# Research log

One entry per working session: what was run, the exact command, the result, and what it means.

## 2026-11-15: code base written and CPU-tested

- Wrote the library, scripts and tests (see README section 8).
- `python -m pytest`: 43 tests pass on CPU (PyTorch 2.15, gsplat 1.5.3, Python 3.13).
- `python scripts/demo_synthetic.py`: both protocols run end to end on the synthetic scene and produce `assets/sandbox_check.png`. Synthetic data with simulated priors; not a result.
- Not yet run: the gsplat CUDA path, VGGT inference, Depth Anything V2 inference, any real dataset.
- Design decisions worth remembering:
  - Scene scale is the extent of the *training* cameras in both protocols, so learning rates and densification thresholds do not depend on held-out cameras.
  - The Sim(3) between the COLMAP frame and the estimated frame uses averaged relative rotations plus the spread of camera centres, because three forward-facing cameras can be nearly collinear and a centres-only (Umeyama) fit is then ill-posed.
  - VGGT input is a pure resize to a 518-pixel longest side with both sides multiples of 14 (no crop or padding), so its intrinsics rescale exactly to the training resolution.
  - Default `init` under COLMAP poses is `sfm_train` (points seen by at least 2 training views). `sfm_all` remains available because it is what many papers use.

## Next

Milestone A in `docs/RUNNING_ON_GPU.md`: full-view 3DGS on LLFF `fern` to validate the GPU path against published numbers.
