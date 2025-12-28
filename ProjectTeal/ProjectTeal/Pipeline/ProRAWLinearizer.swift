//
//  ProRAWLinearizer.swift
//  ProjectTeal
//
//  Produces a linearized CIImage from ProRAW/RAW inputs by disabling local tone mapping
//  and related look-processing, keeping output in a consistent linear space for downstream
//  sensor-consistency checks.
//

import CoreImage
import Foundation

struct ProRAWLinearizer {
    private let contextProvider: CIContextProviding

    init(contextProvider: CIContextProviding = CIContextProvider.shared) {
        self.contextProvider = contextProvider
    }

    /// Attempts to linearize a RAW/ProRAW capture into a tone-mapping-free CGImage.
    /// If the dedicated RAW filter is unavailable, falls back to tone-curve removal.
    /// - Parameters:
    ///   - data: RAW container data from AVCapturePhoto.
    ///   - normalization: Scalar parameters derived from RAW12 tags.
    /// - Returns: Linear RGB CGImage ready for DNG export.
    func makeLinearCGImage(from data: Data, normalization: RAW12Parser.Normalization) -> CGImage? {
        guard let ciImage = makeLinearCIImage(from: data) else {
            return nil
        }

        let normalized = normalization.applying(to: ciImage)
        return contextProvider.renderLinearCGImage(from: normalized, clamped: true)
    }

    private func makeLinearCIImage(from data: Data) -> CIImage? {
        // Try the modern CIRAWFilter API (iOS 15+)
        if let rawFilter = CIRAWFilter(imageData: data, identifierHint: nil) {
            // Disable local tone mapping to keep the output scene-referred/linear
            rawFilter.localToneMapAmount = 0.0
            
            // Disable gamut mapping to preserve original color data
            rawFilter.isGamutMappingEnabled = false
            
            // Disable draft mode for full quality
            rawFilter.isDraftModeEnabled = false
            
            // Set boost to zero to avoid global tone curves
            rawFilter.boostAmount = 0.0
            rawFilter.shadowBias = 0.0
            
            // Disable sharpening and noise reduction to preserve sensor-level detail
            rawFilter.sharpnessAmount = 0.0
            rawFilter.colorNoiseReductionAmount = 0.0
            rawFilter.luminanceNoiseReductionAmount = 0.0
            
            // Disable moire reduction and detail enhancement
            rawFilter.moireReductionAmount = 0.0
            rawFilter.detailAmount = 0.0
            rawFilter.contrastAmount = 0.0

            if let output = rawFilter.outputImage {
                return output
            }
        }

        // Fallback: create CIImage and apply inverse tone curve
        let baseOptions: [CIImageOption: Any] = [
            .applyOrientationProperty: true
        ]
        
        guard let ciImage = CIImage(data: data, options: baseOptions) else {
            return nil
        }

        return ciImage.applyingFilter("CISRGBToneCurveToLinear")
    }
}
