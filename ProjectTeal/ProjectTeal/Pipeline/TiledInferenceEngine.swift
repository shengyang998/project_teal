//
//  TiledInferenceEngine.swift
//  ProjectTeal
//
//  Minimal tiled inference prototype used to validate 48MP memory/latency
//  envelopes. Splits an RGB image into overlapping tiles, runs a provided
//  per-tile processor, and stitches results with cosine/linear blending to
//  suppress seams.
//

import Foundation
import simd

/// Blending profile applied at tile edges to avoid seams.
enum TiledBlendProfile {
    case cosine
    case linear

    func rampUp(_ t: Float) -> Float {
        let clamped = max(0, min(1, t))
        switch self {
        case .cosine:
            let value = cos(Double.pi * Double(clamped))
            return Float(0.5 - 0.5 * value)
        case .linear:
            return clamped
        }
    }

    func rampDown(_ t: Float) -> Float {
        let clamped = max(0, min(1, t))
        switch self {
        case .cosine:
            let value = cos(Double.pi * Double(clamped))
            return Float(0.5 + 0.5 * value)
        case .linear:
            return 1 - clamped
        }
    }
}

/// Configuration for tiled inference.
struct TiledInferenceConfig {
    let tileSize: Int
    let overlap: Int
    let blendProfile: TiledBlendProfile

    init(tileSize: Int = 768, overlap: Int = 48, blendProfile: TiledBlendProfile = .cosine) {
        precondition(tileSize > 0, "Tile size must be positive")
        precondition(overlap >= 0 && overlap * 2 < tileSize, "Overlap must be non-negative and smaller than the tile size")
        self.tileSize = tileSize
        self.overlap = overlap
        self.blendProfile = blendProfile
    }
}

/// Public-facing tile context surfaced to the tile processor.
struct TiledTileContext {
    let originX: Int
    let originY: Int
    let width: Int
    let height: Int
}

/// Describes a tile placement and its overlap constraints.
private struct TilePlacement {
    let context: TiledTileContext
    let leftOverlap: Int
    let rightOverlap: Int
    let topOverlap: Int
    let bottomOverlap: Int
}

/// Executes tiled processing over an RGB image. Intended to wrap Core ML model
/// invocation but accepts any per-tile closure to keep the prototype flexible.
struct TiledInferenceEngine {
    /// Processes an RGB image using the supplied tile processor.
    /// - Parameters:
    ///   - image: Input RGB image.
    ///   - config: Tile sizing and blending configuration.
    ///   - tileProcessor: Closure invoked per tile that must return a tile of the same dimensions.
    /// - Returns: A stitched RGB image.
    func process(image: RGBImage,
                 config: TiledInferenceConfig = TiledInferenceConfig(),
                 tileProcessor: (RGBImage, TiledTileContext) -> RGBImage) -> RGBImage {
        let placements = makePlacements(imageWidth: image.width,
                                        imageHeight: image.height,
                                        config: config)

        var accumPixels = [SIMD3<Float>](repeating: .zero, count: image.width * image.height)
        var accumWeights = [Float](repeating: 0, count: image.width * image.height)

        for placement in placements {
            let crop = crop(image: image, placement: placement)
            let processed = tileProcessor(crop, placement.context)
            precondition(processed.width == placement.context.width && processed.height == placement.context.height,
                         "Tile processor must not change tile dimensions")

            blend(processed: processed,
                  placement: placement,
                  blendProfile: config.blendProfile,
                  accumPixels: &accumPixels,
                  accumWeights: &accumWeights,
                  fullWidth: image.width)
        }

        var outputPixels = [SIMD3<Float>]()
        outputPixels.reserveCapacity(image.width * image.height)

        for index in 0 ..< accumPixels.count {
            let weight = accumWeights[index]
            if weight > 0 {
                outputPixels.append(accumPixels[index] / weight)
            } else {
                outputPixels.append(accumPixels[index])
            }
        }

        return RGBImage(width: image.width, height: image.height, pixels: outputPixels)
    }
}

private extension TiledInferenceEngine {
    func makePlacements(imageWidth: Int, imageHeight: Int, config: TiledInferenceConfig) -> [TilePlacement] {
        let stride = max(1, config.tileSize - config.overlap * 2)
        let xStarts = startPositions(fullLength: imageWidth, tileSize: config.tileSize, stride: stride)
        let yStarts = startPositions(fullLength: imageHeight, tileSize: config.tileSize, stride: stride)

        var placements = [TilePlacement]()
        for yStart in yStarts {
            for xStart in xStarts {
                let width = min(config.tileSize, imageWidth - xStart)
                let height = min(config.tileSize, imageHeight - yStart)
                let leftOverlap = xStart == 0 ? 0 : config.overlap
                let rightOverlap = (xStart + width) >= imageWidth ? 0 : config.overlap
                let topOverlap = yStart == 0 ? 0 : config.overlap
                let bottomOverlap = (yStart + height) >= imageHeight ? 0 : config.overlap

                let context = TiledTileContext(originX: xStart,
                                               originY: yStart,
                                               width: width,
                                               height: height)

                placements.append(TilePlacement(context: context,
                                                leftOverlap: leftOverlap,
                                                rightOverlap: rightOverlap,
                                                topOverlap: topOverlap,
                                                bottomOverlap: bottomOverlap))
            }
        }

        return placements
    }

    func startPositions(fullLength: Int, tileSize: Int, stride: Int) -> [Int] {
        guard fullLength > tileSize else { return [0] }
        var positions: [Int] = [0]
        var current = 0
        while true {
            let next = current + stride
            if next + tileSize >= fullLength {
                positions.append(max(fullLength - tileSize, 0))
                break
            } else {
                positions.append(next)
                current = next
            }
        }

        return Array(Set(positions)).sorted()
    }

    func crop(image: RGBImage, placement: TilePlacement) -> RGBImage {
        var pixels = [SIMD3<Float>](repeating: .zero, count: placement.context.width * placement.context.height)
        for y in 0 ..< placement.context.height {
            for x in 0 ..< placement.context.width {
                let sourceX = placement.context.originX + x
                let sourceY = placement.context.originY + y
                let sourceIndex = sourceY * image.width + sourceX
                let destIndex = y * placement.context.width + x
                pixels[destIndex] = image.pixels[sourceIndex]
            }
        }
        return RGBImage(width: placement.context.width, height: placement.context.height, pixels: pixels)
    }

    func blend(processed: RGBImage,
               placement: TilePlacement,
               blendProfile: TiledBlendProfile,
               accumPixels: inout [SIMD3<Float>],
               accumWeights: inout [Float],
               fullWidth: Int) {
        for y in 0 ..< placement.context.height {
            let weightY = axisWeight(index: y,
                                     length: placement.context.height,
                                     leadingOverlap: placement.topOverlap,
                                     trailingOverlap: placement.bottomOverlap,
                                     profile: blendProfile)
            for x in 0 ..< placement.context.width {
                let weightX = axisWeight(index: x,
                                         length: placement.context.width,
                                         leadingOverlap: placement.leftOverlap,
                                         trailingOverlap: placement.rightOverlap,
                                         profile: blendProfile)
                let weight = weightX * weightY
                let sourceIndex = y * placement.context.width + x
                let destIndex = (placement.context.originY + y) * fullWidth + (placement.context.originX + x)
                accumPixels[destIndex] += processed.pixels[sourceIndex] * SIMD3<Float>(repeating: weight)
                accumWeights[destIndex] += weight
            }
        }
    }

    func axisWeight(index: Int,
                    length: Int,
                    leadingOverlap: Int,
                    trailingOverlap: Int,
                    profile: TiledBlendProfile) -> Float {
        var weight: Float = 1
        if leadingOverlap > 0 && index < leadingOverlap {
            let denominator = max(leadingOverlap - 1, 1)
            let t = Float(index) / Float(denominator)
            weight *= profile.rampUp(t)
        }

        if trailingOverlap > 0 && index >= length - trailingOverlap {
            let positionInOverlap = index - (length - trailingOverlap)
            let denominator = max(trailingOverlap - 1, 1)
            let t = Float(positionInOverlap) / Float(denominator)
            weight *= profile.rampDown(t)
        }

        return weight
    }
}
