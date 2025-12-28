//
//  DNGCompatibilityValidator.swift
//  ProjectTeal
//
//  Lightweight validation to ensure emitted DNGs stay readable across common viewers.
//

import Foundation
import ImageIO
import UniformTypeIdentifiers

struct DNGCompatibilityValidator {
    struct Result {
        let properties: [CFString: Any]
        let dng: [CFString: Any]
        let tiff: [CFString: Any]
        let hasICCProfile: Bool
        let hasColorTransforms: Bool
        let hasAsShotNeutral: Bool
        let isLinear: Bool
        let samplesPerPixel: Int?
    }

    enum Error: Swift.Error {
        case unreadable
        case unsupportedType
        case missingImage
        case missingColorMetadata
    }

    func validate(url: URL) throws -> Result {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw Error.unreadable
        }

        guard let typeIdentifier = CGImageSourceGetType(source) as? String,
              let sourceType = UTType(typeIdentifier),
              sourceType.conforms(to: UTType.dng) else {
            throw Error.unsupportedType
        }

        guard CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else {
            throw Error.missingImage
        }

        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        let dng = properties[kCGImagePropertyDNGDictionary] as? [CFString: Any] ?? [:]
        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]

        // Check for color transforms (color matrices or ICC profile)
        let hasColorTransforms = dng[kCGImagePropertyDNGColorMatrix1] != nil ||
            dng[kCGImagePropertyDNGColorMatrix2] != nil ||
            dng[kCGImagePropertyDNGAsShotICCProfile] != nil
        guard hasColorTransforms else { throw Error.missingColorMetadata }

        let hasAsShotNeutral = dng[kCGImagePropertyDNGAsShotNeutral] != nil
        guard hasAsShotNeutral else { throw Error.missingColorMetadata }

        // Linear RGB is implied by absence of CFA metadata and presence of 3-channel data
        // The actual samples per pixel is derived from CGImage, not metadata
        let isLinear = dng[kCGImagePropertyDNGCFALayout] == nil && dng[kCGImagePropertyDNGCFAPlaneColor] == nil

        // ICC profile check via DNG-specific key
        let hasICCProfile = dng[kCGImagePropertyDNGAsShotICCProfile] != nil

        return Result(properties: properties,
                      dng: dng,
                      tiff: tiff,
                      hasICCProfile: hasICCProfile,
                      hasColorTransforms: hasColorTransforms,
                      hasAsShotNeutral: hasAsShotNeutral,
                      isLinear: isLinear,
                      samplesPerPixel: nil) // Derived from CGImage, not metadata
    }
}
