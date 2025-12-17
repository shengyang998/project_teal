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
        let raw = properties?[kCGImagePropertyRawDictionary] as? [CFString: Any]
        iso = (raw?[kCGImagePropertyExifISOSpeedRatings] as? [Double])?.first
        exposureTimeSeconds = raw?[kCGImagePropertyExifExposureTime] as? Double
        whiteBalance = raw?[kCGImagePropertyExifWhiteBalanceBias] as? [Double]
        blackLevel = raw?[kCGImagePropertyRawBlackLevel] as? Double
        whiteLevel = raw?[kCGImagePropertyRawWhiteLevel] as? Double
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
        if let level = raw[kCGImagePropertyRawBlackLevel] as? Double { return level }
        if let array = raw[kCGImagePropertyRawBlackLevel] as? [Double], let first = array.first { return first }
        if let dngLevel = raw[kCGImagePropertyDNGBlackLevel] as? Double { return dngLevel }
        if let array = raw[kCGImagePropertyDNGBlackLevel] as? [Double], let first = array.first { return first }
        return 0
    }

    private static func whiteLevel(from raw: [CFString: Any]) -> Double {
        if let level = raw[kCGImagePropertyRawWhiteLevel] as? Double { return level }
        if let dngLevel = raw[kCGImagePropertyDNGWhiteLevel] as? Double { return dngLevel }
        return 1
    }

    private static func whiteBalanceGains(from raw: [CFString: Any]) -> SIMD3<Double> {
        if let gains = raw[kCGImagePropertyExifWhiteBalanceBias] as? [Double], gains.count >= 3 {
            return SIMD3(gains[0], gains[1], gains[2])
        }
        if let gains = raw[kCGImagePropertyDNGWhiteBalance] as? [Double], gains.count >= 3 {
            return SIMD3(gains[0], gains[1], gains[2])
        }
        return SIMD3(repeating: 1)
    }

    private static func cfaPattern(from raw: [CFString: Any]) -> CFAPattern {
        guard let pattern = raw[kCGImagePropertyDNGCFAPattern] as? [Int], pattern.count == 4 else {
            return .unknown
        }
        let tuple = (pattern[0], pattern[1], pattern[2], pattern[3])
        switch tuple {
        case (0, 1, 1, 2): return .rggb
        case (2, 1, 1, 0): return .bggr
        case (1, 0, 2, 1): return .grbg
        case (1, 2, 0, 1): return .gbrg
        default: return .unknown
        }
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
