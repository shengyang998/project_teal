//
//  RAWMetadata.swift
//  ProjectTeal
//
//  Lightweight helpers for inspecting RAW captures and computing normalization scalars.
//

import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import simd

struct RAWMetadata {
    let iso: Double?
    let exposureTimeSeconds: Double?
    let whiteBalance: [Double]?
    let blackLevel: Double?
    let whiteLevel: Double?
    let orientation: Int32?
    let pixelSize: CGSize?

    init(imageSource: CGImageSource) {
        let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any]
        let exif = properties?[kCGImagePropertyExifDictionary] as? [CFString: Any]
        let dng = properties?[kCGImagePropertyDNGDictionary] as? [CFString: Any]
        iso = (exif?[kCGImagePropertyExifISOSpeedRatings] as? [Double])?.first
        exposureTimeSeconds = exif?[kCGImagePropertyExifExposureTime] as? Double
        // AsShotNeutral provides RGB neutrals (inverse of WB gains)
        whiteBalance = dng?[kCGImagePropertyDNGAsShotNeutral] as? [Double]
        blackLevel = dng?[kCGImagePropertyDNGBlackLevel] as? Double
        whiteLevel = dng?[kCGImagePropertyDNGWhiteLevel] as? Double
        orientation = properties?[kCGImagePropertyOrientation] as? Int32
        if let width = properties?[kCGImagePropertyPixelWidth] as? CGFloat,
           let height = properties?[kCGImagePropertyPixelHeight] as? CGFloat {
            pixelSize = CGSize(width: width, height: height)
        } else {
            pixelSize = nil
        }
    }
}

/// Extracts RAW12 metadata for normalization and CFA layout inspection.
struct RAW12Parser {
    struct Parsed {
        let blackLevel: Double
        let whiteLevel: Double
        let whiteBalanceGains: SIMD3<Double>
        let cfaPattern: CFAPattern
        let pixelSize: CGSize?

        var normalization: Normalization {
            Normalization(blackLevel: blackLevel,
                          whiteLevel: whiteLevel,
                          whiteBalanceGains: whiteBalanceGains)
        }
    }

    struct Normalization {
        let blackLevel: Double
        let whiteLevel: Double
        let whiteBalanceGains: SIMD3<Double>

        /// Normalize a RAW pixel value into [0, 1].
        func normalize(value: Double) -> Double {
            guard whiteLevel > blackLevel else { return 0 }
            let normalized = (value - blackLevel) / (whiteLevel - blackLevel)
            return normalized.clamped(to: 0 ... 1)
        }

        /// Apply normalization and white balance to a linear CIImage.
        func applying(to image: CIImage) -> CIImage {
            let denominator = max(whiteLevel - blackLevel, Double.leastNonzeroMagnitude)
            let scale = 1.0 / denominator
            let bias = -blackLevel * scale

            let rScale = CGFloat(scale * whiteBalanceGains.x)
            let gScale = CGFloat(scale * whiteBalanceGains.y)
            let bScale = CGFloat(scale * whiteBalanceGains.z)
            let biasVector = CIVector(x: CGFloat(bias), y: CGFloat(bias), z: CGFloat(bias), w: 0)

            return image.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: rScale, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: gScale, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: bScale, w: 0),
                "inputBiasVector": biasVector
            ])
        }
    }

    enum CFAPattern: String {
        case rggb
        case bggr
        case grbg
        case gbrg
        case unknown
    }

    /// Parse RAW12 properties from a CGImageSource.
    func parse(imageSource: CGImageSource) -> Parsed? {
        let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any]
        let rawProps = properties?[kCGImagePropertyRawDictionary] as? [CFString: Any] ?? [:]
        let pixelSize: CGSize?
        if let width = properties?[kCGImagePropertyPixelWidth] as? CGFloat,
           let height = properties?[kCGImagePropertyPixelHeight] as? CGFloat {
            pixelSize = CGSize(width: width, height: height)
        } else {
            pixelSize = nil
        }
        return parse(rawProperties: rawProps, pixelSize: pixelSize)
    }

    /// Parse RAW12 properties from a raw dictionary for testing.
    func parse(rawProperties: [CFString: Any], pixelSize: CGSize?) -> Parsed? {
        let blackLevel = RAW12Parser.blackLevel(from: rawProperties)
        let whiteLevel = RAW12Parser.whiteLevel(from: rawProperties)
        let gains = RAW12Parser.whiteBalanceGains(from: rawProperties)
        let cfaPattern = RAW12Parser.cfaPattern(from: rawProperties)
        return Parsed(blackLevel: blackLevel,
                      whiteLevel: whiteLevel,
                      whiteBalanceGains: gains,
                      cfaPattern: cfaPattern,
                      pixelSize: pixelSize)
    }

    private static func blackLevel(from raw: [CFString: Any]) -> Double {
        // Try DNG black level
        if let dngLevel = raw[kCGImagePropertyDNGBlackLevel] as? Double { return dngLevel }
        if let array = raw[kCGImagePropertyDNGBlackLevel] as? [Double], let first = array.first { return first }
        return 0
    }

    private static func whiteLevel(from raw: [CFString: Any]) -> Double {
        // Try DNG white level
        if let dngLevel = raw[kCGImagePropertyDNGWhiteLevel] as? Double { return dngLevel }
        return 1
    }

    private static func whiteBalanceGains(from raw: [CFString: Any]) -> SIMD3<Double> {
        // AsShotNeutral contains RGB neutral values (inverse of WB gains)
        // To get gains, we invert: gains = 1 / neutral
        if let neutrals = raw[kCGImagePropertyDNGAsShotNeutral] as? [Double], neutrals.count >= 3 {
            let rGain = neutrals[0] > 0 ? 1.0 / neutrals[0] : 1.0
            let gGain = neutrals[1] > 0 ? 1.0 / neutrals[1] : 1.0
            let bGain = neutrals[2] > 0 ? 1.0 / neutrals[2] : 1.0
            // Normalize so green = 1.0
            let normFactor = gGain > 0 ? 1.0 / gGain : 1.0
            return SIMD3(rGain * normFactor, 1.0, bGain * normFactor)
        }
        return SIMD3(repeating: 1)
    }

    private static func cfaPattern(from raw: [CFString: Any]) -> CFAPattern {
        // CFALayout indicates how the CFA is organized
        // For linear RGB (demosaiced) files, there's no CFA pattern
        if let layout = raw[kCGImagePropertyDNGCFALayout] as? Int {
            // CFA layouts: 1=rectangular, 2=staggered A, 3=staggered B, etc.
            // For now, we just detect if CFA metadata exists
            _ = layout // Acknowledge we have CFA data
        }
        // Without the specific pattern array, we can't determine the exact layout
        // Return unknown for demosaiced/linear images
        return .unknown
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
