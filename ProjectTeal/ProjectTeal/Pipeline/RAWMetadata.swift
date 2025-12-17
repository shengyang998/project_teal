//
//  RAWMetadata.swift
//  ProjectTeal
//
//  Lightweight helpers for inspecting RAW captures and computing normalization scalars.
//

import Foundation
import CoreGraphics
import ImageIO

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
        iso = raw?[kCGImagePropertyExifISOSpeedRatings] as? [Double]?.first
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
