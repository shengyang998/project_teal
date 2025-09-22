## Active Context

### Current Focus
- Establish project scaffolding and memory bank.
- Define initial imaging pipeline (RAW open + minimal filter pass).
- Add camera capture to acquire iOS RAW (ProRAW) images.

### Recent Changes
- Initialized Git repository and `.gitignore` for Xcode/CMake.
- Added `MEMORY_BANK.md` and initialized Cursor Memory Bank (`npx cursor-bank init`).
- Implemented `CameraViewController` with preview and ProRAW capture; embedded in root VC.
- Removed Mac Catalyst support; added Mac (designed for iPad) and Vision Pro (designed for iPad) support.

### Next Steps
- Scaffold repo layout and SPM packages.
- Add RxSwift via SPM and set up a shared `CIContext` service.
- Implement RAW open + minimal pipeline (identity + exposure adjust) with exports per stage.
- Create `tools/python/requirements.txt` and `ci_inspect.py` for histograms/side-by-sides.
- Add Info.plist camera and photo library usage descriptions.
- Test camera permissions on Mac (designed for iPad) and Vision Pro (designed for iPad) platforms.

### Resolved Decisions
- **iOS Deployment Target**: iOS 18+ (minimum)
- **UI Framework**: UIKit (not SwiftUI for initial implementation)
- **RAW Formats**: iOS RAW formats only (not DNG or other formats)
- **Color Management**: Display P3 and HDR targets
- **Analysis Workflow**: Exported for human debugging only
- **API Preference**: Use Apple's APIs as preference over third-party alternatives


