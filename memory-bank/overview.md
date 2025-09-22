## Overview (verbatim copy of root MEMORY_BANK.md)

> This file mirrors the current `MEMORY_BANK.md` at the repository root for convenient reference within the memory bank.

## Project Memory Bank

### Purpose
This document captures durable, high-signal decisions and context for the iOS app. It should be kept concise and updated as architecture, tooling, and constraints evolve.

### Vision and Scope
- **Goal**: Build an iOS app that opens RAW images, applies Core Image filters (including custom `CIFilter`s) via a composable filter pipeline, and provides Python tools to analyze/visualize intermediate results.
- **Non-goals (for now)**: Cross-platform UI, non-iOS image acquisition, on-device ML.

### Architecture
- **Style**: Clean Architecture.
  - **Presentation Layer**: MVVM with RxSwift/RxCocoa. ViewModels expose reactive inputs/outputs; no UIKit in ViewModels.
  - **Domain Layer**: Pure business/use-cases. Framework-agnostic. Use-cases orchestrate repositories and services.
  - **Data Layer**: Implementations for persistence, RAW decoding, Core Image contexts, CIFilter pipeline execution, and integrations.
  - **Dependency Rule**: Inner layers have no knowledge of outer layers; boundaries modeled via protocols.
- **Threading/Reactive**: RxSwift for state, user intents, and pipeline progress. Heavy image ops off main thread; UI binding on main thread.
- **Error Handling**: Use typed errors in domain/data; assert preconditions during development. Surface user-relevant failures gracefully.

### Tech Stack and Tooling
- **Language**: Swift (app), Python (analysis tools).
- **UI**: UIKit or SwiftUI (TBD) bound via RxCocoa (if UIKit) or Combine-bridged adapters (if SwiftUI) as needed.
- **Imaging**: Core Image (`CIImage`, `CIFilter`, `CIContext`), RAW via Core Image RAW support (e.g., `CIRAWFilter`). Consider Metal-backed contexts for performance.
- **Package Manager**: Swift Package Manager (SPM) for all third-party Swift dependencies (e.g., RxSwift).
- **Build/Test**: XCTest via Xcode/`xcodebuild`.
  - Test command (SPM/Xcode):
```bash
xcodebuild build-for-testing -scheme catiledmetallayer -destination "platform=iOS Simulator,name=iPhone 16 Pro" -configuration Debug
```
- **Logging & Assertions (dev)**: Prefer `assert`/`precondition` for programmer errors. Consider `os_log`/`Logger` for runtime diagnostics.

### RAW Image Handling (iOS)
- **Acquisition**: Local file URLs, Files app import, or Photos framework (TBD). Prefer URLs for deterministic testing.
- **Decoding**: Use Core Image RAW pipeline (e.g., `CIRAWFilter`) to load sensor data; configure options like exposure, noise reduction, demosaic, color space.
- **Color Management**: Use wide-gamut working space where appropriate; convert to display at presentation time. Ensure consistent `CIContext` `workingColorSpace` and `outputColorSpace`.
- **Performance**: Create shared `CIContext` (Metal-backed if available). Avoid recreating contexts/kernels. Stream with tile or region-of-interest where possible.

### Filter Pipeline Strategy
- **Composable Graph**: Represent pipeline as an ordered list or DAG of filter nodes. Each node receives a `CIImage` and emits a `CIImage` plus metadata.
- **Built-ins**: Use Core Image built-ins where possible for performance and accuracy.
- **Custom Filters**: Implement custom `CIFilter` subclasses with `CIKernel`/Metal-based kernels for hot paths. Define stable parameter keys and validation.
- **Determinism & Reproducibility**: Ensure parameter snapshots, input RAW configuration, and `CIContext` options are captured for exact replays.
- **Introspection**: Emit per-node diagnostics (timings, ROI, parameter hashes) for Python tooling to consume.

### Python Analysis/Visualization Tools
- **Purpose**: Inspect intermediate `CIImage` outputs, compare filter variants, visualize histograms/curves, and validate pipeline correctness/performance.
- **Interchange Format**: Export 16-bit TIFF or PNG per stage, plus a compact JSON manifest of parameters and timings. Consider `.npy` for numeric arrays when precision matters.
- **Directory Plan**: `tools/python/` containing CLI/notebooks.
  - `tools/python/requirements.txt`: `numpy`, `matplotlib`, `opencv-python`, `Pillow`, `tifffile`, optionally `rawpy`.
  - `tools/python/notebooks/` for exploratory analysis.
  - `tools/python/ci_inspect.py` for scripted batch visualization/reporting.
- **Outputs**: `artifacts/analysis/<run-id>/stage-XX-<filter>.tiff` and `manifest.json` for each run.

### Repository Layout (proposed)
- `App/` — Application entry, composition root, DI wiring.
- `Presentation/` — Views, ViewModels (MVVM, RxSwift bindings).
- `Domain/` — Entities, value objects, use-cases, repository protocols.
- `Data/` — Repository impls, Core Image services, RAW loader, file IO.
- `Filters/` — Built-in wrappers, custom `CIFilter`s, pipeline engine.
- `Resources/` — Sample RAWs, color profiles, LUTs.
- `Tools/Python/` — Analysis tools as above.
- `Tests/` — Unit and integration tests per layer.

### Policies and Conventions
- **SOLID**: Prefer protocol-oriented boundaries; open for extension, closed for modification.
- **Interface Segregation**: Small, focused protocols for repositories/services.
- **Preconditions/Assertions**: Use `assert`/`precondition` during development for programmer errors and invariants.
- **Performance**: Favor streaming and GPU-backed contexts; avoid unnecessary intermediate materialization. Memoize kernels.
- **Documentation**: Keep this memory bank concise; add deep technical docs to `docs/` as needed.

### Initial Decisions (ADRs-lite)
1. Use Clean Architecture with MVVM + RxSwift for presentation reactivity and testability.
2. Use SPM exclusively for Swift dependencies to simplify integration and CI.
3. Use Core Image RAW support and `CIFilter` pipeline for image processing; add custom filters where built-ins are insufficient.
4. Provide Python-based analysis tooling for deterministic, scriptable inspection of intermediate pipeline outputs.

### Open Questions
- Minimum iOS deployment target? (e.g., iOS 16/17 for latest Core Image/Metal features)
- UIKit vs SwiftUI for the initial UI?
- Required RAW formats and cameras? (DNG only vs proprietary via iOS decoders)
- Color management targets (Display P3 vs sRGB), and export formats.
- On-device vs exported analysis workflow (where to store artifacts and manifests)?

### Next Steps (suggested)
- Scaffold repo layout and SPM packages.
- Add RxSwift via SPM and set up a shared `CIContext` service.
- Implement RAW open + minimal pipeline (e.g., identity + exposure adjust) with export of per-stage TIFFs.
- Create `tools/python/requirements.txt` and a simple `ci_inspect.py` to render histograms and side-by-sides.


