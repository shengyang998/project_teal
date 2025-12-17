## Active Context

### Current Focus
- Establish project scaffolding and memory bank.
- Define initial imaging pipeline (RAW open + minimal filter pass).
- Add camera capture to acquire iOS RAW (ProRAW) images.
- Lock the 48MP linear RGB plan into executable steps: alignment gates, forward operator tests, and tiled-prototype envelopes.

### Recent Changes
- Initialized Git repository and `.gitignore` for Xcode/CMake.
- Added `MEMORY_BANK.md` and initialized Cursor Memory Bank (`npx cursor-bank init`).
- Implemented `CameraViewController` with preview and ProRAW capture; embedded in root VC.
- Removed Mac Catalyst support; added Mac (designed for iPad) and Vision Pro (designed for iPad) support.
- Captured near-term execution priorities in `ProjectTeal/docs/48mp_linear_rgb_dng_plan.md` to focus alignment, forward-operator, and iOS prototyping work.
- Added a synthetic geometric registration estimator (downsample + cross-correlation) with tests and marked the alignment step complete in `docs/48mp_linear_rgb_dng_steps.md`.

### Next Steps
- Scaffold repo layout and SPM packages.
- Add RxSwift via SPM and set up a shared `CIContext` service.
- Implement RAW open + minimal pipeline (identity + exposure adjust) with exports per stage.
- Create `tools/python/requirements.txt` and `ci_inspect.py` for histograms/side-by-sides.
- Add Info.plist camera and photo library usage descriptions.
- Test camera permissions on Mac (designed for iPad) and Vision Pro (designed for iPad) platforms.
- Swap synthetic alignment tests for fixture-backed thresholds and wire forward-operator unit tests before training.
- Prototype tiled inference without the model to baseline device memory/latency envelopes.

### Resolved Decisions
- **iOS Deployment Target**: iOS 18+ (minimum)
- **UI Framework**: UIKit (not SwiftUI for initial implementation)
- **RAW Formats**: iOS RAW formats only (not DNG or other formats)
- **Color Management**: Display P3 and HDR targets
- **Analysis Workflow**: Exported for human debugging only
- **API Preference**: Use Apple's APIs as preference over third-party alternatives


