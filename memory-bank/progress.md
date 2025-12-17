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

### What's left to build
- Project scaffolding (layers, SPM targets, DI wiring) for iOS 18+.
- iOS RAW ingest and baseline render path using Apple's APIs.
- UIKit UI with MVVM + RxSwift bindings.
- Filter pipeline engine and custom filter scaffolding.
- Display P3/HDR color management implementation.
- Python analysis tools and export pipeline for debugging.
- Execution of the 48MP linear RGB DNG step plan in `ProjectTeal/docs/48mp_linear_rgb_dng_steps.md`.
- Alignment fixtures + forward-operator tests to gate model training.
- Early tiled-inference prototype (no model) to validate device memory/latency envelopes for 48MP frames.

### Current status
- Early setup phase; requirements and architecture documented.
- Camera view controller implemented and embedded; ProRAW capture path in place.
- Key decisions resolved: iOS 18+, UIKit, iOS RAW only, Display P3/HDR, Apple API preference.
- Near-term plan clarified with alignment/forward-operator gates before training and tiled-prototype validation on device.

### Known issues
- Simulator has no camera; preview/capture require device testing.
