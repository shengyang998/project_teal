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
        return contextProvider.renderLinearCGImage(from: normalized)
    }

    private func makeLinearCIImage(from data: Data) -> CIImage? {
        let baseOptions: [CIImageOption: Any] = [
            .applyOrientationProperty: true,
            .cache: false
        ]

        if let rawFilter = CIRAWFilter(imageData: data, options: baseOptions) {
            // Disable local tone mapping and gamut mapping to keep the output scene-referred.
            rawFilter.setValue(true, forKey: kCIInputDisableLocalToneMap as String)
            rawFilter.setValue(true, forKey: kCIInputDisableGamutMap as String)
            rawFilter.setValue(false, forKey: kCIInputAllowDraftModeKey as String)

            // Bias/boost set to zero keeps the decoder from applying global tone curves.
            rawFilter.setValue(0.0, forKey: kCIInputBoostKey as String)
            rawFilter.setValue(0.0, forKey: kCIInputBiasKey as String)
            rawFilter.setValue(true, forKey: "inputLinearSpaceFilter")

            // Avoid default sharpening/noise reduction that could mask sensor-consistency issues.
            rawFilter.setValue(false, forKey: kCIInputEnableSharpening as String)
            rawFilter.setValue(false, forKey: kCIInputEnableChromaticNoiseReduction as String)

            if let output = rawFilter.outputImage {
                return output
            }
        }

        guard let ciImage = CIImage(data: data, options: baseOptions) else {
            return nil
        }

        return ciImage.applyingFilter("CISRGBToneCurveToLinear")
    }
}
