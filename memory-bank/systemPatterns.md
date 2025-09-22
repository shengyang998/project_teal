## System Architecture and Patterns

### Architecture Style
- Clean Architecture with MVVM in the presentation layer.
- Layers: Presentation (MVVM, RxSwift), Domain (use-cases, entities), Data (repositories, services).
- Dependency rule: inner layers independent of frameworks; communicate via protocols.

### Presentation
- ViewModels expose reactive inputs/outputs via RxSwift/RxCocoa.
- No UIKit references inside ViewModels.
- UI bindings on main thread; heavy work off main thread.

### Domain
- Pure Swift; framework-agnostic.
- Use-cases coordinate repositories/services and encapsulate business rules.

### Data
- Repositories implement abstractions for RAW decoding, Core Image processing, IO.
- `CIContext` is shared (prefer Metal-backed). Avoid recreation during a session.

### Filter Pipeline Pattern
- Ordered nodes (or DAG) each mapping `CIImage -> CIImage` with metadata.
- Prefer built-in filters; add custom `CIFilter` subclasses where needed.
- Capture deterministic parameters and environment for reproducibility.
- Emit diagnostics: timings, ROI, parameter hashes per node.

### Repository/Layout Conventions
- `App/` composition root and DI
- `Presentation/` views + view models
- `Domain/` entities, value objects, use-cases, repository protocols
- `Data/` implementations, CI services, RAW loader, file IO
- `Filters/` built-in wrappers, custom filters, pipeline engine
- `Resources/` sample RAWs, color profiles, LUTs
- `Tests/` unit/integration

### Policies and Conventions
- SOLID with interface segregation; protocol-first boundaries.
- Use `assert`/`precondition` for programmer errors during development.
- Favor GPU-backed processing, avoid unnecessary materialization, memoize kernels.
- Keep docs concise; deep dives live in `docs/`.

### Initial Decisions (ADRs-lite)
1. Clean Architecture + MVVM with RxSwift.
2. Swift Package Manager for all Swift dependencies.
3. Core Image RAW and `CIFilter` pipeline with custom filters as needed.
4. Python tools for deterministic stage inspection and analysis.
5. iOS 18+ minimum deployment target with UIKit UI.
6. iOS RAW formats only (not DNG or other formats).
7. Display P3 and HDR color management targets.
8. Prefer Apple's APIs over third-party alternatives.


