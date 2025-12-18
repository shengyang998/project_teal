## Active Context

### Current Focus
- Lock the 48MP linear RGB plan into executable steps with sensor-consistency gates.
- Stand up the forward operator and synthetic tests that gate training against the 12MP RAW anchor.
- Prepare for gain-field/differentiable model scaffolding and export hooks.

### Recent Changes
- Initialized Git repository and `.gitignore` for Xcode/CMake.
- Added `MEMORY_BANK.md` and initialized Cursor Memory Bank (`npx cursor-bank init`).
- Implemented `CameraViewController` with preview and ProRAW capture; embedded in root VC.
- Removed Mac Catalyst support; added Mac (designed for iPad) and Vision Pro (designed for iPad) support.
- Captured near-term execution priorities in `ProjectTeal/docs/48mp_linear_rgb_dng_plan.md` to focus alignment, forward-operator, and iOS prototyping work.
- Added a synthetic geometric registration estimator (downsample + cross-correlation) with tests and marked the alignment step complete in `docs/48mp_linear_rgb_dng_steps.md`.
- Introduced a fixture-backed regression test for the geometric alignment estimator to enforce translation/score thresholds.
- Implemented the quad-Bayer forward operator with per-channel scaling and 2×2 binning plus synthetic pattern tests; marked the sensor-consistency step complete in `docs/48mp_linear_rgb_dng_steps.md`.

### Next Steps
- Wire model scaffolding for gain-field/differentiable heads and export hooks.
- Add anchor and gradient losses that consume the forward operator output.
- Build tiled inference prototype (no model) to validate device memory/latency envelopes on target hardware.
- Keep Python/CI analysis utilities (histograms/side-by-sides) in sync with RAW ingest paths.

### Resolved Decisions
- **iOS Deployment Target**: iOS 18+ (minimum)
- **UI Framework**: UIKit (not SwiftUI for initial implementation)
- **RAW Formats**: iOS RAW formats only (not DNG or other formats)
- **Color Management**: Display P3 and HDR targets
- **Analysis Workflow**: Exported for human debugging only
- **API Preference**: Use Apple's APIs as preference over third-party alternatives


