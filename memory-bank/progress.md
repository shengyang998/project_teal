## Progress

### What works
- Git initialized; `.gitignore` configured for Xcode/CMake/general dev artifacts.
- Memory bank created with core files.
- SwiftUI entry `ProjectTealApp` -> `AppRootView` -> `AppRootViewController`.
- Live camera preview via `AVCaptureVideoPreviewLayer`.
- Photo capture with Apple ProRAW on supported devices; falls back otherwise.
- Mac (designed for iPad) and Vision Pro (designed for iPad) support enabled; Mac Catalyst support removed.

### What's left to build
- Project scaffolding (layers, SPM targets, DI wiring) for iOS 18+.
- Add Info.plist privacy keys for camera and photo library permissions.
- iOS RAW ingest and baseline render path using Apple's APIs.
- UIKit UI with MVVM + RxSwift bindings.
- Filter pipeline engine and custom filter scaffolding.
- Display P3/HDR color management implementation.
- Python analysis tools and export pipeline for debugging.

### Current status
- Early setup phase; requirements and architecture documented.
- Camera view controller implemented and embedded; ProRAW capture path in place.
- Key decisions resolved: iOS 18+, UIKit, iOS RAW only, Display P3/HDR, Apple API preference.

### Known issues
- Simulator has no camera; preview/capture require device testing.


