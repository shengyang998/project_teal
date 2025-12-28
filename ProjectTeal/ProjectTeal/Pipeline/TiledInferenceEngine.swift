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

/// Configuration for the low-resolution gain-field pass.
struct TiledGainFieldConfig {
    /// Integer downsample factor used to shrink the input before gain prediction.
    let downsample: Int
    /// Tile sizing and blending parameters applied in the low-res domain.
    let tiling: TiledInferenceConfig

    init(downsample: Int = 4, tiling: TiledInferenceConfig = TiledInferenceConfig(tileSize: 640,
                                                                                  overlap: 48,
                                                                                  blendProfile: .cosine)) {
        precondition(downsample > 0, "Downsample factor must be positive")
        self.downsample = downsample
        self.tiling = tiling
    }
}

/// Captures latency and working-set estimates for a tiled inference pass.
struct TiledInferenceMetrics {
    let tileCount: Int
    let tileDurations: [TimeInterval]
    let gainTileDurations: [TimeInterval]
    let totalDuration: TimeInterval
    let accumulatorBytes: Int
    let gainAccumulatorBytes: Int
    let maxTileBytes: Int
    let maxGainTileBytes: Int

    /// Estimated peak memory usage during inference (accumulators + active tiles).
    var estimatedWorkingSetBytes: Int {
        let accumulator = max(accumulatorBytes, gainAccumulatorBytes)
        return accumulator + maxTileBytes + maxGainTileBytes
    }

    var averageTileDuration: TimeInterval {
        guard !tileDurations.isEmpty else { return 0 }
        return tileDurations.reduce(0, +) / Double(tileDurations.count)
    }

    var averageGainTileDuration: TimeInterval {
        guard !gainTileDurations.isEmpty else { return 0 }
        return gainTileDurations.reduce(0, +) / Double(gainTileDurations.count)
    }
}

/// Bundles the stitched image with its performance metrics.
struct TiledInferenceResult {
    let image: RGBImage
    let metrics: TiledInferenceMetrics
}

/// Public-facing tile context surfaced to the tile processor.
struct TiledTileContext {
    let originX: Int
    let originY: Int
    let width: Int
    let height: Int
}

/// Captures the low-res gain field covering a single high-res tile.
struct GainFieldTile {
    let width: Int
    let height: Int
    let pixels: [SIMD3<Float>]

    init(width: Int, height: Int, pixels: [SIMD3<Float>]) {
        precondition(pixels.count == width * height, "Gain tile pixel count must match dimensions")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    subscript(x: Int, y: Int) -> SIMD3<Float> {
        pixels[y * width + x]
    }
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
        processWithMetrics(image: image, config: config, tileProcessor: tileProcessor).image
    }

    /// Processes an RGB image and returns performance metrics for latency/memory budgeting.
    func processWithMetrics(image: RGBImage,
                            config: TiledInferenceConfig = TiledInferenceConfig(),
                            tileProcessor: (RGBImage, TiledTileContext) -> RGBImage) -> TiledInferenceResult {
        let placements = makePlacements(imageWidth: image.width,
                                        imageHeight: image.height,
                                        config: config)

        var accumPixels = [SIMD3<Float>](repeating: .zero, count: image.width * image.height)
        var accumWeights = [Float](repeating: 0, count: image.width * image.height)

        var tileDurations: [TimeInterval] = []
        tileDurations.reserveCapacity(placements.count)

        var maxTileBytes = 0
        let accumulatorBytes = estimatedAccumulatorBytes(width: image.width, height: image.height)
        let startTime = CFAbsoluteTimeGetCurrent()

        for placement in placements {
            maxTileBytes = max(maxTileBytes, estimatedTileBytes(width: placement.context.width, height: placement.context.height))

            let crop = crop(image: image, placement: placement)
            let tileStart = CFAbsoluteTimeGetCurrent()
            let processed = tileProcessor(crop, placement.context)
            let tileEnd = CFAbsoluteTimeGetCurrent()
            tileDurations.append(tileEnd - tileStart)

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

        let totalDuration = CFAbsoluteTimeGetCurrent() - startTime
        let metrics = TiledInferenceMetrics(tileCount: placements.count,
                                            tileDurations: tileDurations,
                                            gainTileDurations: [],
                                            totalDuration: totalDuration,
                                            accumulatorBytes: accumulatorBytes,
                                            gainAccumulatorBytes: 0,
                                            maxTileBytes: maxTileBytes,
                                            maxGainTileBytes: 0)

        return TiledInferenceResult(image: RGBImage(width: image.width, height: image.height, pixels: outputPixels),
                                    metrics: metrics)
    }

    /// Processes an RGB image with an auxiliary low-resolution gain-field pass.
    /// - Parameters:
    ///   - image: Input RGB image.
    ///   - config: Tile sizing and blending configuration for the high-res pass.
    ///   - gainConfig: Downsampling and tiling configuration for the gain-field pass.
    ///   - gainProcessor: Closure invoked per low-res tile that returns per-channel gains.
    ///   - tileProcessor: Closure invoked per high-res tile with the corresponding gain tile.
    /// - Returns: A stitched RGB image.
    func processWithGainField(image: RGBImage,
                              config: TiledInferenceConfig = TiledInferenceConfig(),
                              gainConfig: TiledGainFieldConfig = TiledGainFieldConfig(),
                              gainProcessor: (RGBImage, TiledTileContext) -> RGBImage,
                              tileProcessor: (RGBImage, TiledTileContext, GainFieldTile) -> RGBImage) -> RGBImage {
        processWithGainFieldAndMetrics(image: image,
                                       config: config,
                                       gainConfig: gainConfig,
                                       gainProcessor: gainProcessor,
                                       tileProcessor: tileProcessor).image
    }

    /// Processes an RGB image with a low-resolution gain-field pass and collects performance metrics.
    func processWithGainFieldAndMetrics(image: RGBImage,
                                        config: TiledInferenceConfig = TiledInferenceConfig(),
                                        gainConfig: TiledGainFieldConfig = TiledGainFieldConfig(),
                                        gainProcessor: (RGBImage, TiledTileContext) -> RGBImage,
                                        tileProcessor: (RGBImage, TiledTileContext, GainFieldTile) -> RGBImage) -> TiledInferenceResult {
        let lowResImage = downsample(image: image, factor: gainConfig.downsample)
        let lowResResult = processWithMetrics(image: lowResImage,
                                              config: gainConfig.tiling,
                                              tileProcessor: gainProcessor)
        let fullResGain = upsampleNearest(image: lowResResult.image,
                                          factor: gainConfig.downsample,
                                          targetWidth: image.width,
                                          targetHeight: image.height)

        let placements = makePlacements(imageWidth: image.width,
                                        imageHeight: image.height,
                                        config: config)

        var accumPixels = [SIMD3<Float>](repeating: .zero, count: image.width * image.height)
        var accumWeights = [Float](repeating: 0, count: image.width * image.height)

        var tileDurations: [TimeInterval] = []
        tileDurations.reserveCapacity(placements.count)

        var maxTileBytes = 0
        var maxGainTileBytes = max(lowResResult.metrics.maxTileBytes, 0)

        let accumulatorBytes = estimatedAccumulatorBytes(width: image.width, height: image.height)
        let startTime = CFAbsoluteTimeGetCurrent()

        for placement in placements {
            let tileBytes = estimatedTileBytes(width: placement.context.width, height: placement.context.height)
            maxTileBytes = max(maxTileBytes, tileBytes)
            maxGainTileBytes = max(maxGainTileBytes, tileBytes)

            let tileCrop = crop(image: image, placement: placement)
            let gainCrop = crop(image: fullResGain, placement: placement)
            let gainTile = GainFieldTile(width: gainCrop.width, height: gainCrop.height, pixels: gainCrop.pixels)

            let tileStart = CFAbsoluteTimeGetCurrent()
            let processed = tileProcessor(tileCrop, placement.context, gainTile)
            let tileEnd = CFAbsoluteTimeGetCurrent()
            tileDurations.append(tileEnd - tileStart)

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

        let totalDuration = (CFAbsoluteTimeGetCurrent() - startTime) + lowResResult.metrics.totalDuration
        let metrics = TiledInferenceMetrics(tileCount: placements.count,
                                            tileDurations: tileDurations,
                                            gainTileDurations: lowResResult.metrics.tileDurations,
                                            totalDuration: totalDuration,
                                            accumulatorBytes: accumulatorBytes,
                                            gainAccumulatorBytes: lowResResult.metrics.accumulatorBytes,
                                            maxTileBytes: maxTileBytes,
                                            maxGainTileBytes: maxGainTileBytes)

        return TiledInferenceResult(image: RGBImage(width: image.width, height: image.height, pixels: outputPixels),
                                    metrics: metrics)
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

    func downsample(image: RGBImage, factor: Int) -> RGBImage {
        guard factor > 1 else { return image }
        let outputWidth = Int(ceil(Double(image.width) / Double(factor)))
        let outputHeight = Int(ceil(Double(image.height) / Double(factor)))
        var outputPixels = [SIMD3<Float>](repeating: .zero, count: outputWidth * outputHeight)

        for y in 0 ..< outputHeight {
            for x in 0 ..< outputWidth {
                var sum = SIMD3<Float>(repeating: 0)
                var count: Float = 0

                for dy in 0 ..< factor {
                    for dx in 0 ..< factor {
                        let sourceX = x * factor + dx
                        let sourceY = y * factor + dy
                        if sourceX < image.width && sourceY < image.height {
                            sum += image[sourceX, sourceY]
                            count += 1
                        }
                    }
                }

                outputPixels[y * outputWidth + x] = count > 0 ? sum / count : sum
            }
        }

        return RGBImage(width: outputWidth, height: outputHeight, pixels: outputPixels)
    }

    func upsampleNearest(image: RGBImage, factor: Int, targetWidth: Int, targetHeight: Int) -> RGBImage {
        guard factor > 1 else { return image }
        var pixels = [SIMD3<Float>](repeating: .zero, count: targetWidth * targetHeight)

        for y in 0 ..< targetHeight {
            let sourceY = min(y / factor, image.height - 1)
            for x in 0 ..< targetWidth {
                let sourceX = min(x / factor, image.width - 1)
                let source = image[sourceX, sourceY]
                pixels[y * targetWidth + x] = source
            }
        }

        return RGBImage(width: targetWidth, height: targetHeight, pixels: pixels)
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

    func estimatedTileBytes(width: Int, height: Int) -> Int {
        width * height * MemoryLayout<SIMD3<Float>>.stride
    }

    func estimatedAccumulatorBytes(width: Int, height: Int) -> Int {
        let pixelBytes = width * height * MemoryLayout<SIMD3<Float>>.stride
        let weightBytes = width * height * MemoryLayout<Float>.stride
        return pixelBytes + weightBytes
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
