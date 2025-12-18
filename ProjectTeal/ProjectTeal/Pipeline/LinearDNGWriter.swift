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

struct LinearDNGWriter {
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
    ///   - destinationURL: URL to write the DNG file.
    ///   - options: Writer configuration for tiling/compression.
    func write(cgImage: CGImage,
               metadata: [CFString: Any]?,
               destinationURL: URL,
               options: Options = Options()) throws -> Output {
        guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL,
                                                                 UTType.digitalNegative.identifier as CFString,
                                                                 1,
                                                                 nil) else {
            throw Error.cannotCreateDestination
        }

        let properties = makeDestinationProperties(metadata: metadata, options: options) as CFDictionary?
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
    func makeDestinationProperties(metadata: [CFString: Any]?, options: Options) -> [CFString: Any] {
        var properties = metadata ?? [:]

        var dng = (properties[kCGImagePropertyDNGDictionary] as? [CFString: Any]) ?? [:]
        let cfaKeys: [CFString] = [
            kCGImagePropertyDNGCFAPattern,
            kCGImagePropertyDNGCFAPlaneColor,
            kCGImagePropertyDNGCFARepeatPatternDim,
            kCGImagePropertyDNGCFALayout
        ]
        cfaKeys.forEach { dng.removeValue(forKey: $0) }
        dng[kCGImagePropertyDNGIsLinearRaw] = true
        properties[kCGImagePropertyDNGDictionary] = dng

        var tiff = (properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any]) ?? [:]
        tiff[kCGImagePropertyTIFFSamplesPerPixel] = 3
        tiff[kCGImagePropertyTIFFCompression] = options.compression.tiffValue

        if let tileSize = options.tileSize {
            tiff[kCGImagePropertyTIFFTileWidth] = tileSize
            tiff[kCGImagePropertyTIFFTileLength] = tileSize
        } else {
            tiff.removeValue(forKey: kCGImagePropertyTIFFTileWidth)
            tiff.removeValue(forKey: kCGImagePropertyTIFFTileLength)
        }

        properties[kCGImagePropertyTIFFDictionary] = tiff
        return properties
    }
}
