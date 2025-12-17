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
    func write(cgImage: CGImage, metadata: [CFString: Any]?, destinationURL: URL) throws -> Output {
        guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL,
                                                                 UTType.digitalNegative.identifier as CFString,
                                                                 1,
                                                                 nil) else {
            throw Error.cannotCreateDestination
        }

        let options = metadata as CFDictionary?
        CGImageDestinationAddImage(destination, cgImage, options)

        guard CGImageDestinationFinalize(destination) else {
            throw Error.failedFinalize
        }

        let attributes = try FileManager.default.attributesOfItem(atPath: destinationURL.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        return Output(url: destinationURL, fileSize: size)
    }
}
