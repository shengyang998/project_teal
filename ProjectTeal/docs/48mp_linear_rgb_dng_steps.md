# 48MP Linear RGB DNG Step-by-Step Implementation

This guide breaks down the plan in `48mp_linear_rgb_dng_plan.md` into concrete, traceable execution steps. Each step is scoped to be independently testable and should land via small PRs.

## 0. Environment & Repo Scaffolding
- [x] Add Swift Package Manager dependencies (e.g., RxSwift) and shared utilities target.
- [x] Standardize lint/format settings and CI tasks.
- [x] Create a shared `CIContext` service and dependency-injection entry point.

## 1. Data Ingest & Alignment
- [x] Implement RAW12 parsing: black/white level, CFA layout, WB gains, normalization to linear.
- [x] Linearize ProRAW48 capture (undo EOTF, avoid local ops) into consistent linear space.
- [x] Geometric registration: compute 2× scale + translation/warp; validate by downsampling ProRAW48 to 12MP and comparing to demosaiced RAW12 edges (synthetic harness in place).
- [x] Add fixture-backed unit tests that assert alignment error thresholds (synthetic correlation harness landed).

## 2. Differentiable Sensor-Consistency Operator
- [x] Implement forward operator `F`: mosaic predicted 48MP RGB into quad-Bayer, 2×2 bin to 12MP, apply per-channel scaling.
- [x] Optimize for speed and determinism; keep the graph differentiable.
- [x] Add synthetic-pattern tests to verify indexing and binning correctness.

## 3. Model Architecture
- [x] Build two-head UNet-style network with Core ML–friendly ops.
- [x] Inputs: linearized ProRAW48 (detail) + RAW12 guidance (demosaic → 12MP RGB → 2× upsample).
- [x] Outputs: gain field head (smooth inverse-LTM gains) and detail head (full-res residual); compose \`\hat{L}_{48} = max(0, I^{lin}_{48} \odot g + r)\`.
- [x] Provide model export hooks (Torch → Core ML) and shape tests.

## 4. Losses & Training Schedule
- [x] Anchor loss: Charbonnier on \`F(\hat{L}_{48})\` vs RAW12 mosaic using the differentiable forward operator with optional per-channel scaling.
- [x] Detail preservation: gradient loss after global-only rendering (tone curve + CCM; no local ops).
- [x] Gain regularization: |∇g|₁ smoothness + range penalties.
- [x] Training schedule: anchor-heavy start; introduce gradient/detail loss once anchor stabilizes; track configs (loss weights, gain range, tone curve/CCM) alongside checkpoints.

## 5. Evaluation Harness
- [ ] Metrics: PSNR/SSIM on RAW12 mosaic, per-CFA error histograms, gradient correlation/halo indicators post global rendering.
- [ ] Curate qualitative set (backlit, mixed light, bright windows, city lights, foliage, fine patterns).
- [ ] Implement report generator comparing to global inverse tone-curve baseline.

## 6. iOS Inference Pipeline
- [ ] Tiled inference (512–1024 px tiles, 32–64 px overlap) with cosine/linear blending.
- [ ] Low-res gain field over larger context; stitch into full-res output.
- [ ] Core ML integration with supported ops and FP16 weights; evaluate INT8 after visual sign-off.
- [ ] Ensure latency/memory targets for 48MP capture.

## 7. 48MP Linear RGB DNG Writer
- [ ] Emit 3-channel linear image (tiled, optional lossless compression), SamplesPerPixel=3 (no CFA tags).
- [ ] Write metadata: color matrices/ICC, AsShotNeutral/WB, baseline exposure/black level matching normalization.
- [ ] Compatibility validation in Lightroom/ACR, Apple Photos/Preview, and an open-source reader.

## 8. Baselines & Risk Mitigation
- [ ] Implement Baseline A: global de-LTM mapping from ProRAW48 to RAW12-guided linear exposure.
- [ ] Implement Baseline B: gain-field-only model for quick LTM removal.
- [ ] Add early tests for misalignment, quad/binner correctness, and WB consistency.

## Acceptance Gates (Definition of Done)
- Each step includes automated tests or analysis notebooks with documented thresholds.
- RAW-anchor consistency validated by re-mosaic + 2×2 binning check against 12MP RAW.
- Visual checks show reduced local tone-mapping artifacts with stable color and preserved detail.
- iOS pipeline writes a compatible 48MP linear RGB DNG within device performance budgets.
