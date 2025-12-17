//
//  CIContextProvider.swift
//  ProjectTeal
//
//  Shared CIContext utilities for RAW processing.
//

import CoreImage
import CoreGraphics

/// Provides a shared CIContext configured for linear output suitable for RAW pipelines.
struct CIContextProvider {
    static let shared = CIContextProvider()

    /// Lazily created CIContext bound to a linear color space.
    let context: CIContext = {
        let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB)
        let options: [CIContextOption: Any] = [
            .workingColorSpace: colorSpace as Any,
            .outputColorSpace: colorSpace as Any,
            .cacheIntermediates: false
        ]
        return CIContext(options: options)
    }()

    /// Renders the given CIImage into a 16-bit-per-channel CGImage with linear space.
    /// - Parameters:
    ///   - image: Input CIImage assumed to be in linear space or tagged appropriately.
    ///   - clamped: If true, clamps extent before rendering to avoid partial tiles from filters.
    func renderLinearCGImage(from image: CIImage, clamped: Bool = true) -> CGImage? {
        let target = clamped ? image.clampedToExtent() : image
        let destinationColorSpace = CGColorSpace(name: CGColorSpace.linearSRGB)
        return context.createCGImage(target, from: target.extent, format: .RGBAh, colorSpace: destinationColorSpace, deferred: false)
            ?? context.createCGImage(target, from: target.extent, format: .RGBAf, colorSpace: destinationColorSpace, deferred: false)
    }
}
