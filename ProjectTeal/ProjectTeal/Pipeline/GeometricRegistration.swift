//
//  GeometricRegistration.swift
//  ProjectTeal
//
//  Estimates the geometric relationship between a 48MP ProRAW frame and a
//  12MP RAW anchor by downsampling and computing translation offsets that
//  maximize normalized cross-correlation. The scale is derived from input
//  dimensions (expected to be ~2× for quad-Bayer captures).
//

import Foundation
import simd

/// Minimal grayscale buffer used for alignment math without platform image APIs.
struct GrayscaleImage {
    let width: Int
    let height: Int
    var pixels: [Float]

    init(width: Int, height: Int, pixels: [Float]) {
        precondition(pixels.count == width * height, "Pixel count must match dimensions")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    subscript(x: Int, y: Int) -> Float {
        get {
            guard (0 ..< width).contains(x), (0 ..< height).contains(y) else { return 0 }
            return pixels[y * width + x]
        }
        set {
            guard (0 ..< width).contains(x), (0 ..< height).contains(y) else { return }
            pixels[y * width + x] = newValue
        }
    }
}

/// Reports the estimated geometric relationship between 48MP and 12MP frames.
struct GeometricAlignmentResult: Equatable {
    let scale: SIMD2<Double>
    let translation: SIMD2<Double>
    let score: Double
}

/// Computes translation (and implied scale) between a high-res and low-res pair.
struct GeometricRegistration {
    /// Downsamples the high-res frame to low-res and searches for the translation
    /// that maximizes normalized cross-correlation within a bounded window.
    /// - Parameters:
    ///   - highRes: Typically the 48MP linearized frame.
    ///   - lowRes: Typically the 12MP demosaiced anchor.
    ///   - searchRadius: Half-window for integer translation search in low-res pixels.
    func estimateAlignment(highRes: GrayscaleImage,
                           lowRes: GrayscaleImage,
                           searchRadius: Int = 4) -> GeometricAlignmentResult {
        let downsampled = downsample2x(highRes)
        let translation = bestCorrelationOffset(reference: lowRes,
                                                candidate: downsampled,
                                                searchRadius: searchRadius)
        let scale = SIMD2<Double>(Double(highRes.width) / Double(lowRes.width),
                                  Double(highRes.height) / Double(lowRes.height))
        return GeometricAlignmentResult(scale: scale,
                                        translation: SIMD2(Double(translation.x), Double(translation.y)),
                                        score: translation.score)
    }

    private func downsample2x(_ image: GrayscaleImage) -> GrayscaleImage {
        let targetWidth = image.width / 2
        let targetHeight = image.height / 2
        var pixels = [Float](repeating: 0, count: targetWidth * targetHeight)

        for y in 0 ..< targetHeight {
            for x in 0 ..< targetWidth {
                let sx = x * 2
                let sy = y * 2
                let sum = image[sx, sy] + image[sx + 1, sy] + image[sx, sy + 1] + image[sx + 1, sy + 1]
                pixels[y * targetWidth + x] = sum * 0.25
            }
        }

        return GrayscaleImage(width: targetWidth, height: targetHeight, pixels: pixels)
    }

    private func bestCorrelationOffset(reference: GrayscaleImage,
                                       candidate: GrayscaleImage,
                                       searchRadius: Int) -> (x: Int, y: Int, score: Double) {
        var bestScore = -Double.infinity
        var bestOffset = (x: 0, y: 0)

        for dy in -searchRadius ... searchRadius {
            for dx in -searchRadius ... searchRadius {
                let score = correlation(reference: reference, candidate: candidate, offsetX: dx, offsetY: dy)
                if score > bestScore {
                    bestScore = score
                    bestOffset = (dx, dy)
                }
            }
        }

        return (bestOffset.x, bestOffset.y, bestScore)
    }

    private func correlation(reference: GrayscaleImage,
                             candidate: GrayscaleImage,
                             offsetX: Int,
                             offsetY: Int) -> Double {
        let xStart = max(0, -offsetX)
        let yStart = max(0, -offsetY)
        let xEnd = min(reference.width, candidate.width - offsetX)
        let yEnd = min(reference.height, candidate.height - offsetY)

        guard xEnd > xStart, yEnd > yStart else { return -Double.infinity }

        var sumRef: Double = 0
        var sumCand: Double = 0
        var sumRefSq: Double = 0
        var sumCandSq: Double = 0
        var sumProduct: Double = 0
        var count: Int = 0

        for y in yStart ..< yEnd {
            for x in xStart ..< xEnd {
                let ref = Double(reference[x, y])
                let cand = Double(candidate[x + offsetX, y + offsetY])
                sumRef += ref
                sumCand += cand
                sumRefSq += ref * ref
                sumCandSq += cand * cand
                sumProduct += ref * cand
                count += 1
            }
        }

        guard count > 0 else { return -Double.infinity }

        let invCount = 1.0 / Double(count)
        let meanRef = sumRef * invCount
        let meanCand = sumCand * invCount

        let numerator = sumProduct - Double(count) * meanRef * meanCand
        let denomLeft = sumRefSq - Double(count) * meanRef * meanRef
        let denomRight = sumCandSq - Double(count) * meanCand * meanCand
        let denominator = sqrt(max(denomLeft * denomRight, 0))

        guard denominator > 0 else { return -Double.infinity }
        return numerator / denominator
    }
}
