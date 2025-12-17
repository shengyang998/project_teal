# 48MP Linear RGB DNG Pipeline Plan

This document captures a practical plan to ship a 48MP linear RGB DNG that removes ProRAW local tone mapping and other look-processing while remaining physically consistent with the 12MP RAW binned anchor.

## Output Contract & Acceptance Tests
- Deliver a 48MP scene-referred linear RGB DNG (16-bit integer or float) rendered via standard DNG metadata.
- Acceptance checks:
  1. RAW-anchor consistency: re-mosaic + 2×2 bin of the 48MP output should tightly match the 12MP RAW mosaic (L1/PSNR in linear domain).
  2. No local tone-mapping artifacts: halos reduced; global tone consistent across boundaries.
  3. Detail retention: edges/textures comparable to ProRAW48 without over-smoothing.
  4. Color stability: white balance and chroma remain stable across the frame.

Deliverables include preprocessing, trainable model + training code, iOS inference (tiling + stitching), and a linear DNG writer.

## Phase 1 — Data Pipeline and Alignment
- Parse RAW12: read BlackLevel/WhiteLevel/CFA tags, normalize to linear, apply WB gains in mosaic domain.
- Linearize ProRAW48: undo gamma/EOTF, avoid local ops, keep in a consistent linear space.
- Geometric registration: compute scale (2×), translation, and small warp per capture mode/device. Validate by downsampling ProRAW48 to 12MP and checking edges against demosaiced RAW12.

## Phase 2 — Differentiable Sensor-Consistency Operator
- Define forward operator F: mosaic the predicted 48MP linear RGB into quad-Bayer, 2×2 bin to 12MP, apply any per-channel scaling. Must be differentiable, exact, and fast.
- Provide unit tests with synthetic patterns to verify indexing and correctness.

## Phase 3 — Model Design
- Inputs: linearized ProRAW48 (detail carrier) and RAW12 guidance (demosaic → 12MP RGB → 2× upsample).
- Architecture: two-head UNet-style network.
  - Gain field head predicts smooth inverse-LTM gain map at low-res.
  - Detail head predicts small residual at full-res.
- Composition: \hat{L}_{48} = max(0, I^{lin}_{48} ⊙ g + r).
- Deliver Torch model with Core ML–friendly ops.

## Phase 4 — Losses and Training
- Primary anchor loss: L1/Charbonnier on F(\hat{L}_{48}) vs RAW12 mosaic.
- Detail preservation: gradient loss after global-only rendering (tone curve + CCM; no local ops).
- Gain regularization: smoothness (|∇g|_1) and range penalties.
- Schedule: start anchor-heavy; introduce detail term after convergence.

## Phase 5 — Evaluation
- Quantitative: PSNR/SSIM on RAW12 mosaic; per-CFA error histograms; gradient correlation and halo indicators after global rendering.
- Qualitative set: backlit scenes, mixed lighting, bright windows, city lights, foliage, fine patterns.
- Deliver report comparing to a global inverse tone-curve baseline.

## Phase 6 — iOS Deployment
- Tiling: 512–1024 px tiles with 32–64 px overlap; blend with cosine/linear weights; run gain field at low-res over larger context if possible.
- Core ML: use supported ops, FP16 weights; consider INT8 after visual sign-off.
- Deliver on-device pipeline processing 48MP within latency/memory budget.

## Phase 7 — Write 48MP Linear RGB DNG
- Structure: 3-channel linear image, tiled, optionally lossless compressed; SamplesPerPixel=3 (no CFA tags).
- Metadata: color matrices or ICC, AsShotNeutral/WB, baseline exposure/black level aligned to normalization.
- Compatibility: Lightroom/ACR, Apple Photos/Preview, and an open-source reader.

## Phase 8 — Baselines and Risk Mitigation
- Baseline A: global de-LTM mapping from ProRAW48 to RAW12-guided linear exposure for fallback.
- Baseline B: gain-field-only model to quickly remove LTM.
- Key risks: misalignment, incorrect quad/binner logic, WB ambiguity—address early with tests and standardized metadata.

## Recommended Execution Order
1. Finish alignment and correct forward operator (Phases 1–2).
2. Train gain-only model with anchor loss + mild regularization.
3. Add gradient detail loss; evaluate halos/texture.
4. Add residual head if needed.
5. Integrate tiled inference on iOS.
6. Finalize linear DNG writer and compatibility tests.
