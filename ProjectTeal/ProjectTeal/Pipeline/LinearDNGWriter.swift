//
//  LinearDNGWriter.swift
//  ProjectTeal
//
//  Writes a linear RGB CGImage into a DNG container with basic metadata for debugging compatibility.
//

import Foundation
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics
import simd

struct LinearDNGWriter {
    private static let linearSRGBToXYZ: [Double] = [
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    ]

    static let identity3x3: [Double] = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]

    struct Options {
        enum Compression {
            case none
            case losslessJPEG

            var tiffValue: Int {
                switch self {
                case .none: return 1 // Uncompressed
                case .losslessJPEG: return 34712 // Adobe-style lossless JPEG
                }
            }
        }

        /// Tile edge in pixels. When nil, strips tiling hints and lets ImageIO pick defaults.
        let tileSize: Int?
        /// Lossless compression flag; DNG supports JPEG-lossless for RGB data.
        let compression: Compression

        init(tileSize: Int? = nil, compression: Compression = .none) {
            self.tileSize = tileSize
            self.compression = compression
        }
    }

    struct Output {
        let url: URL
        let fileSize: UInt64
    }

    enum Error: Swift.Error {
        case cannotCreateDestination
        case failedFinalize
    }

    /// Writes a linear RGB image to disk as a DNG with optional metadata.
    /// - Parameters:
    ///   - cgImage: Linear RGB image to embed.
    ///   - metadata: Image metadata dictionary, typically from RAW source.
    ///   - normalization: Black level/white level/WB gains used to produce the linearized image.
    ///   - destinationURL: URL to write the DNG file.
    ///   - options: Writer configuration for tiling/compression.
    func write(cgImage: CGImage,
               metadata: [CFString: Any]?,
               normalization: RAW12Parser.Normalization? = nil,
               destinationURL: URL,
               options: Options = Options()) throws -> Output {
        guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL,
                                                                 UTType.dng.identifier as CFString,
                                                                 1,
                                                                 nil) else {
            throw Error.cannotCreateDestination
        }

        let properties = makeDestinationProperties(metadata: metadata,
                                                  normalization: normalization,
                                                  options: options) as CFDictionary?
        CGImageDestinationAddImage(destination, cgImage, properties)

        guard CGImageDestinationFinalize(destination) else {
            throw Error.failedFinalize
        }

        let attributes = try FileManager.default.attributesOfItem(atPath: destinationURL.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        return Output(url: destinationURL, fileSize: size)
    }

    /// Builds a sanitized metadata dictionary for a 3-channel linear RGB DNG.
    /// Exposed internally for testing.
    func makeDestinationProperties(metadata: [CFString: Any]?,
                                   normalization: RAW12Parser.Normalization? = nil,
                                   options: Options) -> [CFString: Any] {
        var properties = metadata ?? [:]

        var dng = (properties[kCGImagePropertyDNGDictionary] as? [CFString: Any]) ?? [:]
        let cfaKeys: [CFString] = [
            kCGImagePropertyDNGCFAPlaneColor,
            kCGImagePropertyDNGCFALayout
        ]
        cfaKeys.forEach { dng.removeValue(forKey: $0) }
        // Linear RGB is implied by removing CFA metadata and setting SamplesPerPixel=3
        //         dng[kCGImagePropertyDNGIsLinearRaw] = true

        LinearDNGWriter.populateNormalizationMetadata(into: &dng, normalization: normalization)
        LinearDNGWriter.populateColorMetadata(into: &dng)

        properties[kCGImagePropertyDNGDictionary] = dng

        var tiff = (properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any]) ?? [:]
        // SamplesPerPixel is automatically derived from the CGImage (RGB = 3)
        tiff[kCGImagePropertyTIFFCompression] = options.compression.tiffValue

        if let tileSize = options.tileSize {
            tiff[kCGImagePropertyTIFFTileWidth] = tileSize
            tiff[kCGImagePropertyTIFFTileLength] = tileSize
        } else {
            tiff.removeValue(forKey: kCGImagePropertyTIFFTileWidth)
            tiff.removeValue(forKey: kCGImagePropertyTIFFTileLength)
        }

        properties[kCGImagePropertyTIFFDictionary] = tiff

        if properties[kCGImagePropertyColorModel] == nil {
            properties[kCGImagePropertyColorModel] = kCGImagePropertyColorModelRGB
        }

        if properties[kCGImagePropertyProfileName] == nil {
            properties[kCGImagePropertyProfileName] = "Linear sRGB"
        }
        return properties
    }

    private static func makeAsShotNeutral(from gains: SIMD3<Double>) -> [Double] {
        [
            1.0 / gains.x,
            1.0 / gains.y,
            1.0 / gains.z
        ]
    }

    private static func populateNormalizationMetadata(into dng: inout [CFString: Any],
                                                      normalization: RAW12Parser.Normalization?) {
        if let normalization {
            dng[kCGImagePropertyDNGBlackLevel] = Array(repeating: 0.0, count: 3)
            dng[kCGImagePropertyDNGWhiteLevel] = 1.0
            dng[kCGImagePropertyDNGAsShotNeutral] = Self.makeAsShotNeutral(from: normalization.whiteBalanceGains)
            dng[kCGImagePropertyDNGBaselineExposure] = 0.0
        } else {
            if dng[kCGImagePropertyDNGBlackLevel] == nil {
                dng[kCGImagePropertyDNGBlackLevel] = Array(repeating: 0.0, count: 3)
            }
            if dng[kCGImagePropertyDNGWhiteLevel] == nil {
                dng[kCGImagePropertyDNGWhiteLevel] = 1.0
            }
            if dng[kCGImagePropertyDNGAsShotNeutral] == nil {
                dng[kCGImagePropertyDNGAsShotNeutral] = [1.0, 1.0, 1.0]
            }
            if dng[kCGImagePropertyDNGBaselineExposure] == nil {
                dng[kCGImagePropertyDNGBaselineExposure] = 0.0
            }
        }
    }

    private static func populateColorMetadata(into dng: inout [CFString: Any]) {
        if dng[kCGImagePropertyDNGColorMatrix1] == nil {
            dng[kCGImagePropertyDNGColorMatrix1] = linearSRGBToXYZ
        }
        if dng[kCGImagePropertyDNGColorMatrix2] == nil {
            dng[kCGImagePropertyDNGColorMatrix2] = linearSRGBToXYZ
        }
        if dng[kCGImagePropertyDNGCameraCalibration1] == nil {
            dng[kCGImagePropertyDNGCameraCalibration1] = identity3x3
        }
        if dng[kCGImagePropertyDNGCameraCalibration2] == nil {
            dng[kCGImagePropertyDNGCameraCalibration2] = identity3x3
        }
        if dng[kCGImagePropertyDNGAsShotNeutral] == nil {
            dng[kCGImagePropertyDNGAsShotNeutral] = [1.0, 1.0, 1.0]
        }
        if dng[kCGImagePropertyDNGCalibrationIlluminant1] == nil {
            dng[kCGImagePropertyDNGCalibrationIlluminant1] = 21 // D65
        }
        if dng[kCGImagePropertyDNGCalibrationIlluminant2] == nil {
            dng[kCGImagePropertyDNGCalibrationIlluminant2] = 21 // D65
        }
        // Embed linear sRGB ICC profile for color-managed applications
        if dng[kCGImagePropertyDNGAsShotICCProfile] == nil,
           let iccData = CGColorSpace(name: CGColorSpace.linearSRGB)?.copyICCData() as Data? {
            dng[kCGImagePropertyDNGAsShotICCProfile] = iccData
        }
    }
}
