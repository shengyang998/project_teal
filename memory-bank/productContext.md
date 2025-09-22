## Product Context

### Why this project exists
Photographers and developers need a precise, reproducible way to open RAW images on iOS and apply a tunable pipeline of image processing operations, while preserving data fidelity and enabling deep inspection.

### Problems it solves
- Lack of transparent, stage-by-stage visibility into image processing on-device.
- Difficulty validating and comparing filter parameters and sequences.
- Fragmented tooling between mobile processing and desktop analysis.

### How it should work
- User selects a RAW file; the app decodes it using Core Image RAW support.
- A configurable pipeline applies built-in and custom `CIFilter`s.
- The app can export intermediate images and a manifest of parameters/timings for analysis.

### User experience goals
- Responsive UI with progress indication while processing.
- Clear controls for pipeline configuration and parameter presets.
- Easy export/sharing of per-stage outputs for offline analysis.


