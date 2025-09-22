## Technical Context

### Languages and Frameworks
- Swift (iOS app) - iOS 18+ minimum deployment target
- Python (analysis tooling)
- UIKit for UI (not SwiftUI for initial implementation)
- Core Image: `CIImage`, `CIFilter`, `CIContext`, `CIRAWFilter`
- Metal-backed `CIContext` where available
- RxSwift/RxCocoa for reactivity

### Package Management
- Swift Package Manager (SPM) for all Swift dependencies (e.g., RxSwift).

### Build and Test
- XCTest for unit/integration tests.
- Command (SPM/Xcode):
```bash
xcodebuild build-for-testing -scheme catiledmetallayer -destination "platform=iOS Simulator,name=iPhone 16 Pro" -configuration Debug
```

### Imaging and Color Management
- Use Core Image RAW pipeline for iOS RAW formats only (not DNG or other formats).
- Configure exposure, noise reduction, demosaic, color spaces via Apple's APIs.
- Use Display P3 and HDR color management targets.
- Prefer wide-gamut working color space; convert to display at presentation time.
- Keep `workingColorSpace`/`outputColorSpace` consistent.

### Python Tooling
- Directory: `tools/python/`
- Suggested requirements: `numpy`, `matplotlib`, `opencv-python`, `Pillow`, `tifffile`, optional `rawpy`.
- Exports: 16-bit TIFF/PNG per stage + JSON manifest.
- Purpose: Human debugging and analysis only (not production workflow).

### Constraints and Considerations
- Performance: share `CIContext`, avoid recreation; move heavy ops off main thread.
- Determinism: snapshot parameters and CI context options for reproducibility.
- Logging/Assertions: prefer `assert`/`precondition` for dev; `os_log`/`Logger` for runtime.
- API Preference: Use Apple's APIs as preference over third-party alternatives.


