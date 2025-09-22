## Project Brief

### Core Goal
Build an iOS 18+ app that opens iOS RAW images, applies Core Image filters (including custom `CIFilter`s) via a composable filter pipeline, and provides Python tools to analyze and visualize intermediate results for human debugging.

### Scope
- iOS RAW image ingest via Apple's Core Image RAW support (iOS 18+).
- UIKit-based UI with MVVM + RxSwift architecture.
- Filter pipeline with built-in and custom filters using Apple's APIs.
- Display P3 and HDR color management.
- Deterministic exports per stage for human debugging analysis.
- Python-based analysis/visualization tooling.

### Non-Goals (initial phase)
- Cross-platform UI.
- Non-iOS image acquisition pipelines.
- On-device ML-based enhancement.

### Success Criteria
- Can open common RAW formats (e.g., DNG) and render baseline output.
- Pipeline supports composition, parameterization, and reproducible runs.
- Per-stage artifacts (TIFF/PNG + JSON manifest) generated for analysis.
- App architecture remains testable and maintainable (Clean Architecture + MVVM).


