## Active Context

### Current Focus
- Lock the 48MP linear RGB plan into executable steps with sensor-consistency gates.
- Build out evaluation and reporting around the curated qualitative set and quantitative metrics.
- Maintain alignment between on-device ingest paths and Python analysis utilities.

### Recent Changes
- Initialized Git repository and `.gitignore` for Xcode/CMake.
- Added `MEMORY_BANK.md` and initialized Cursor Memory Bank (`npx cursor-bank init`).
- Implemented `CameraViewController` with preview and ProRAW capture; embedded in root VC.
- Removed Mac Catalyst support; added Mac (designed for iPad) and Vision Pro (designed for iPad) support.
- Captured near-term execution priorities in `ProjectTeal/docs/48mp_linear_rgb_dng_plan.md` to focus alignment, forward-operator, and iOS prototyping work.
- Added a synthetic geometric registration estimator (downsample + cross-correlation) with tests and marked the alignment step complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Introduced a fixture-backed regression test for the geometric alignment estimator to enforce translation/score thresholds.
- Implemented the quad-Bayer forward operator with per-channel scaling and 2×2 binning plus synthetic pattern tests; marked the sensor-consistency step complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Built a PyTorch two-head UNet scaffold with gain-field and detail heads plus Core ML export helper and shape tests.
- Added input/guidance preparation and output composition helpers around the gain-field model with unit tests; marked the model-input/output steps complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Added a Core ML tracing helper with shape introspection and tests, and marked the export step complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Implemented anchor/gradient/gain regularization losses that consume the forward operator output with targeted unit tests and marked step 4 complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Added evaluation metrics for PSNR/SSIM on the RAW mosaic, per-CFA residual histograms, and gradient correlation/halo indicators after global rendering, with unit tests; marked the metrics step complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Curated qualitative evaluation coverage (backlit, mixed light, bright windows, city lights, foliage, fine patterns) with a manifest loader and fixture-backed tests; marked the qualitative-set step complete in `docs/48mp_linear_rgb_dng_steps.md`.

### Next Steps
- Build tiled inference prototype (no model) to validate device memory/latency envelopes on target hardware.
- Add evaluation reporting that consumes the curated qualitative set and quantitative metrics for side-by-side regressions.
- Keep Python/CI analysis utilities (histograms/side-by-sides) in sync with RAW ingest paths.

### Resolved Decisions
- **iOS Deployment Target**: iOS 18+ (minimum)
- **UI Framework**: UIKit (not SwiftUI for initial implementation)
- **RAW Formats**: iOS RAW formats only (not DNG or other formats)
- **Color Management**: Display P3 and HDR targets
- **Analysis Workflow**: Exported for human debugging only
- **API Preference**: Use Apple's APIs as preference over third-party alternatives


