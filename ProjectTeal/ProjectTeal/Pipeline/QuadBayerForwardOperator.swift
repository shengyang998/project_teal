//
//  QuadBayerForwardOperator.swift
//  ProjectTeal
//
//  Implements the differentiable-friendly forward operator that maps a
//  predicted 48MP linear RGB image to the 12MP RAW anchor domain by
//  applying quad-Bayer mosaicing followed by 2×2 binning. This mirrors the
//  device binner so losses can be computed directly against the RAW mosaic.

import Foundation
import simd

/// Simple RGB container used for synthetic forward-operator tests.
struct RGBImage {
    let width: Int
    let height: Int
    var pixels: [SIMD3<Float>]

    init(width: Int, height: Int, pixels: [SIMD3<Float>]) {
        precondition(pixels.count == width * height, "Pixel count must match dimensions")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    subscript(x: Int, y: Int) -> SIMD3<Float> {
        get {
            pixels[y * width + x]
        }
        set {
            pixels[y * width + x] = newValue
        }
    }
}

/// Represents a mosaiced image in the RAW anchor domain.
struct MosaicImage {
    let width: Int
    let height: Int
    let cfaPattern: RAW12Parser.CFAPattern
    var pixels: [Float]

    init(width: Int, height: Int, cfaPattern: RAW12Parser.CFAPattern, pixels: [Float]) {
        precondition(pixels.count == width * height, "Pixel count must match dimensions")
        self.width = width
        self.height = height
        self.cfaPattern = cfaPattern
        self.pixels = pixels
    }

    subscript(x: Int, y: Int) -> Float {
        get {
            pixels[y * width + x]
        }
        set {
            pixels[y * width + x] = newValue
        }
    }
}

/// Applies the forward operator used by the training loss: mosaics a predicted
/// 48MP RGB image into quad-Bayer space and bins it down to the 12MP RAW anchor.
struct QuadBayerForwardOperator {
    /// Mosaics and bins the provided RGB image.
    /// - Parameters:
    ///   - image: High-resolution RGB image (e.g., predicted 48MP linear output).
    ///   - cfaPattern: Quad-Bayer CFA pattern at the block level (defaults to RGGB).
    ///   - channelScale: Optional per-channel scaling applied before binning.
    /// - Returns: A mosaiced 12MP image matching the RAW anchor layout.
    func apply(to image: RGBImage,
               cfaPattern: RAW12Parser.CFAPattern = .rggb,
               channelScale: SIMD3<Float> = SIMD3<Float>(repeating: 1)) -> MosaicImage {
        precondition(image.width % 2 == 0 && image.height % 2 == 0, "Input dimensions must be even for 2x2 binning")

        let outputWidth = image.width / 2
        let outputHeight = image.height / 2
        var outputPixels = [Float](repeating: 0, count: outputWidth * outputHeight)

        for y in 0 ..< outputHeight {
            for x in 0 ..< outputWidth {
                var sum: Float = 0
                let baseX = x * 2
                let baseY = y * 2

                for dy in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        let hx = baseX + dx
                        let hy = baseY + dy
                        let channelIndex = QuadBayerForwardOperator.channelIndex(forX: hx,
                                                                                y: hy,
                                                                                pattern: cfaPattern)
                        let pixel = image[hx, hy]
                        let scaled = pixel[channelIndex] * channelScale[channelIndex]
                        sum += scaled
                    }
                }

                outputPixels[y * outputWidth + x] = sum * 0.25
            }
        }

        return MosaicImage(width: outputWidth,
                           height: outputHeight,
                           cfaPattern: cfaPattern,
                           pixels: outputPixels)
    }

    /// Maps a quad-Bayer coordinate to its channel index using 2×2 color blocks.
    private static func channelIndex(forX x: Int, y: Int, pattern: RAW12Parser.CFAPattern) -> Int {
        let blockX = (x / 2) & 1
        let blockY = (y / 2) & 1

        switch pattern {
        case .rggb:
            if blockX == 0 && blockY == 0 { return 0 }
            if blockX == 1 && blockY == 0 { return 1 }
            if blockX == 0 && blockY == 1 { return 1 }
            return 2
        case .bggr:
            if blockX == 0 && blockY == 0 { return 2 }
            if blockX == 1 && blockY == 0 { return 1 }
            if blockX == 0 && blockY == 1 { return 1 }
            return 0
        case .grbg:
            if blockX == 0 && blockY == 0 { return 1 }
            if blockX == 1 && blockY == 0 { return 0 }
            if blockX == 0 && blockY == 1 { return 2 }
            return 1
        case .gbrg:
            if blockX == 0 && blockY == 0 { return 1 }
            if blockX == 1 && blockY == 0 { return 2 }
            if blockX == 0 && blockY == 1 { return 0 }
            return 1
        case .unknown:
            return 1
        }
    }
}
