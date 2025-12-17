//
//  AppEnvironment.swift
//  ProjectTeal
//
//  Centralized dependency container for reusable services.
//

import Foundation
import RxRelay

/// Shared, lightweight services used throughout the app.
struct AppEnvironment {
    static let shared = AppEnvironment()

    let ciContextProvider: CIContextProviding
    let rawPipeline: RAWProcessingPipeline
    let captureStatus: CaptureStatusStore

    init(ciContextProvider: CIContextProviding = CIContextProvider.shared,
         rawPipeline: RAWProcessingPipeline? = nil,
         captureStatus: CaptureStatusStore = CaptureStatusStore()) {
        self.ciContextProvider = ciContextProvider
        self.rawPipeline = rawPipeline ?? RAWProcessingPipeline(contextProvider: ciContextProvider)
        self.captureStatus = captureStatus
    }
}

/// Observable store for capture status messages.
final class CaptureStatusStore {
    let status = BehaviorRelay<String>(value: "Ready for RAW capture")

    func update(_ message: String) {
        status.accept(message)
    }
}
