## Progress

### What works
- Git initialized; `.gitignore` configured for Xcode/CMake/general dev artifacts.
- Memory bank created with core files.
- SwiftUI entry `ProjectTealApp` -> `AppRootView` -> `AppRootViewController`.
- Live camera preview via `AVCaptureVideoPreviewLayer`.
- Photo capture with Apple ProRAW on supported devices; falls back otherwise.
- Baseline RAW ingest pipeline that linearizes captures and emits a debuggable DNG artifact.
- Mac (designed for iPad) and Vision Pro (designed for iPad) support enabled; Mac Catalyst support removed.
- Privacy keys for camera and photo library add access configured via build settings.
- Synthetic geometric registration estimator (2× downsample + cross-correlation) with unit tests to validate translation recovery.
- Fixture-backed regression test that asserts translation and correlation thresholds for the alignment estimator.
- Quad-Bayer forward operator that mosaics 48MP RGB into CFA space, 2×2 bins to 12MP, and applies per-channel scaling with synthetic tests for indexing/binning correctness.
- Two-head PyTorch UNet scaffold for gain-field and detail heads with a Core ML tracing helper (output shapes) and unit tests.
- Model input/output helpers that upsample RAW12 guidance, concatenate inputs, and compose gain+residual outputs with unit tests.
- PyTorch training losses for the anchor forward operator (Charbonnier), gradient/detail preservation after global rendering, and gain-field smoothness/range regularization with unit tests.
- Evaluation metrics covering PSNR/SSIM on the mosaiced RAW anchor, per-CFA residual histograms, and gradient correlation/halo indicators after global rendering, with unit tests.
- Qualitative manifest loader and fixture-backed coverage for backlit, mixed light, bright windows, city lights, foliage, and fine-pattern scenarios with unit tests.
- Reporting helper that compares candidate metrics against a global inverse tone-curve baseline and surfaces qualitative tag coverage.
- Tiled inference prototype with configurable tiles/overlaps, cosine/linear blending, and unit tests to validate stitching and coverage.
- Low-res gain-field pass that downsamples, tiles, blends, upsamples, and injects gains into the high-res tiles with unit validation.

### What's left to build
- Project scaffolding (layers, SPM targets, DI wiring) for iOS 18+.
- iOS RAW ingest and baseline render path using Apple's APIs.
- UIKit UI with MVVM + RxSwift bindings.
- Filter pipeline engine and custom filter scaffolding.
- Display P3/HDR color management implementation.
- Python analysis tools and export pipeline for debugging.
- Execution of the 48MP linear RGB DNG step plan in `ProjectTeal/docs/48mp_linear_rgb_dng_steps.md`.
- Evaluation harness reporting layer: qualitative coverage is curated and metrics exist; extend reporting outputs into the on-device pipeline.

### Current status
- Early setup phase; requirements and architecture documented.
- Camera view controller implemented and embedded; ProRAW capture path in place.
- Key decisions resolved: iOS 18+, UIKit, iOS RAW only, Display P3/HDR, Apple API preference.
- Near-term plan clarified with alignment/forward-operator gates before training and tiled-prototype validation on device.

### Known issues
- Simulator has no camera; preview/capture require device testing.
