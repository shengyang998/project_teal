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
        case missingLinearFlag
        case missingColorMetadata
        case invalidSamplesPerPixel
    }

    func validate(url: URL) throws -> Result {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw Error.unreadable
        }

        guard let type = CGImageSourceGetType(source),
              UTTypeConformsTo(type, UTType.digitalNegative.identifier as CFString) else {
            throw Error.unsupportedType
        }

        guard CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else {
            throw Error.missingImage
        }

        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        let dng = properties[kCGImagePropertyDNGDictionary] as? [CFString: Any] ?? [:]
        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]

        let isLinear = (dng[kCGImagePropertyDNGIsLinearRaw] as? Bool) == true
        guard isLinear else { throw Error.missingLinearFlag }

        let hasColorTransforms = dng[kCGImagePropertyDNGColorMatrix1] != nil ||
            dng[kCGImagePropertyDNGColorMatrix2] != nil ||
            properties[kCGImagePropertyICCProfile] != nil
        guard hasColorTransforms else { throw Error.missingColorMetadata }

        let hasAsShotNeutral = dng[kCGImagePropertyDNGAsShotNeutral] != nil
        guard hasAsShotNeutral else { throw Error.missingColorMetadata }

        let samplesPerPixel = tiff[kCGImagePropertyTIFFSamplesPerPixel] as? Int
        guard samplesPerPixel == 3 else { throw Error.invalidSamplesPerPixel }

        let hasICCProfile = (properties[kCGImagePropertyICCProfile] as? Data)?.isEmpty == false

        return Result(properties: properties,
                      dng: dng,
                      tiff: tiff,
                      hasICCProfile: hasICCProfile,
                      hasColorTransforms: hasColorTransforms,
                      hasAsShotNeutral: hasAsShotNeutral,
                      isLinear: isLinear,
                      samplesPerPixel: samplesPerPixel)
    }
}
