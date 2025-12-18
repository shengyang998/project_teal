//
//  TiledInferenceEngineTests.swift
//  ProjectTealTests
//
//  Validates tile placement, blending, and stitching behavior for the tiled
//  inference prototype. Ensures coverage, normalization, and seam softening are
//  stable for 48MP-scale tiling configs.
//

import Foundation
@testable import ProjectTeal
import Testing

struct TiledInferenceEngineTests {
    @Test func preservesIdentityAcrossTiles() throws {
        // Create a small synthetic image with distinct per-channel ramps.
        let width = 8
        let height = 6
        var pixels = [SIMD3<Float>]()
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = Float(y * width + x)
                pixels.append(SIMD3<Float>(base, base * 0.25, base * 0.5))
            }
        }
        let image = RGBImage(width: width, height: height, pixels: pixels)

        let engine = TiledInferenceEngine()
        let output = engine.process(image: image,
                                    config: TiledInferenceConfig(tileSize: 4, overlap: 1, blendProfile: .cosine)) { tile, _ in
            tile // identity processor
        }

        #expect(output.width == image.width)
        #expect(output.height == image.height)

        for index in 0 ..< pixels.count {
            #expect(output.pixels[index].isApproximatelyEqual(to: pixels[index], tolerance: 1e-4))
        }
    }

    @Test func blendsOverlapsWithLinearRamp() throws {
        // Two horizontal tiles with overlap; tile processors emit constant values
        // so the overlap should interpolate between them.
        let width = 8
        let height = 2
        let image = RGBImage(width: width,
                             height: height,
                             pixels: [SIMD3<Float>](repeating: SIMD3<Float>(repeating: 0), count: width * height))

        let engine = TiledInferenceEngine()
        let config = TiledInferenceConfig(tileSize: 6, overlap: 2, blendProfile: .linear)
        let output = engine.process(image: image, config: config) { tile, context in
            // Encode tile origin into the output so blending is visible.
            let value: Float = context.originX == 0 ? 0 : 2
            return RGBImage(width: tile.width,
                            height: tile.height,
                            pixels: [SIMD3<Float>](repeating: SIMD3<Float>(repeating: value),
                                                   count: tile.width * tile.height))
        }

        // Expected grid: two tiles (0..5) and (2..7) with overlap columns 2..5.
        // Column 2 should lean to the left tile; column 4 should be an even blend.
        func outputValue(x: Int) -> Float {
            let index = x // height is 2 so row 0 is fine
            return output.pixels[index].x
        }

        let column2 = outputValue(x: 2)
        let column4 = outputValue(x: 4)
        let column6 = outputValue(x: 6)

        #expect(column2 < 1.0) // closer to left tile's 0
        #expect(column4.isApproximatelyEqual(to: 1.0, tolerance: 0.05)) // blended center
        #expect(column6 > 1.5) // dominated by right tile's value
    }

    @Test func stitchesLowResGainFieldIntoHighResTiles() throws {
        // Build a small image and force tiling on both the gain-field and high-res passes.
        let width = 6
        let height = 4
        let image = RGBImage(width: width,
                             height: height,
                             pixels: [SIMD3<Float>](repeating: SIMD3<Float>(repeating: 1), count: width * height))

        let engine = TiledInferenceEngine()
        let gainDownsample = 2
        let lowResWidth = Int(ceil(Double(width) / Double(gainDownsample)))
        let gainConfig = TiledGainFieldConfig(downsample: gainDownsample,
                                             tiling: TiledInferenceConfig(tileSize: 3, overlap: 1, blendProfile: .linear))

        let output = engine.processWithGainField(image: image,
                                                 config: TiledInferenceConfig(tileSize: 4, overlap: 1, blendProfile: .cosine),
                                                 gainConfig: gainConfig,
                                                 gainProcessor: { _, context in
                                                     var pixels = [SIMD3<Float>]()
                                                     for y in 0 ..< context.height {
                                                         for x in 0 ..< context.width {
                                                             let globalX = context.originX + x
                                                             let globalY = context.originY + y
                                                             let value = Float(globalY * lowResWidth + globalX)
                                                             pixels.append(SIMD3<Float>(repeating: value))
                                                         }
                                                     }
                                                     return RGBImage(width: context.width, height: context.height, pixels: pixels)
                                                 },
                                                 tileProcessor: { tile, _, gainTile in
                                                     var pixels = [SIMD3<Float>]()
                                                     for y in 0 ..< tile.height {
                                                         for x in 0 ..< tile.width {
                                                             pixels.append(tile[x, y] * gainTile[x, y])
                                                         }
                                                     }
                                                     return RGBImage(width: tile.width, height: tile.height, pixels: pixels)
                                                 })

        #expect(output.width == width)
        #expect(output.height == height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let lowX = x / gainDownsample
                let lowY = y / gainDownsample
                let expected = Float(lowY * lowResWidth + lowX)
                let pixel = output[x, y]
                #expect(pixel.x.isApproximatelyEqual(to: expected, tolerance: 1e-4))
                #expect(pixel.y.isApproximatelyEqual(to: expected, tolerance: 1e-4))
                #expect(pixel.z.isApproximatelyEqual(to: expected, tolerance: 1e-4))
            }
        }
    }

    @Test func surfacesLatencyAndMemoryMetrics() throws {
        // Validate that metrics track both the low-res gain tiles and the high-res tiles.
        let width = 8
        let height = 8
        let image = RGBImage(width: width,
                             height: height,
                             pixels: [SIMD3<Float>](repeating: SIMD3<Float>(repeating: 1), count: width * height))

        let engine = TiledInferenceEngine()
        let result = engine.processWithGainFieldAndMetrics(image: image,
                                                           config: TiledInferenceConfig(tileSize: 4, overlap: 1, blendProfile: .cosine),
                                                           gainConfig: TiledGainFieldConfig(downsample: 2,
                                                                                            tiling: TiledInferenceConfig(tileSize: 3,
                                                                                                                        overlap: 1,
                                                                                                                        blendProfile: .linear)),
                                                           gainProcessor: { tile, _ in tile },
                                                           tileProcessor: { tile, _, gainTile in
                                                               var pixels = [SIMD3<Float>]()
                                                               pixels.reserveCapacity(tile.width * tile.height)
                                                               for index in 0 ..< tile.pixels.count {
                                                                   pixels.append(tile.pixels[index] * gainTile.pixels[index])
                                                               }
                                                               return RGBImage(width: tile.width, height: tile.height, pixels: pixels)
                                                           })

        let metrics = result.metrics
        #expect(metrics.tileCount == 9) // 3x3 high-res tiles with overlap stride
        #expect(metrics.gainTileDurations.count == 4) // 2x2 low-res gain tiles
        #expect(metrics.accumulatorBytes > 0)
        #expect(metrics.maxTileBytes > 0)
        #expect(metrics.maxGainTileBytes >= metrics.maxTileBytes)
        #expect(metrics.estimatedWorkingSetBytes >= metrics.accumulatorBytes + metrics.maxTileBytes)
        #expect(metrics.totalDuration >= 0)
    }
}

private extension SIMD3 where Scalar == Float {
    func isApproximatelyEqual(to other: SIMD3<Float>, tolerance: Float) -> Bool {
        let dx = abs(x - other.x)
        let dy = abs(y - other.y)
        let dz = abs(z - other.z)
        return dx <= tolerance && dy <= tolerance && dz <= tolerance
    }
}

private extension Float {
    func isApproximatelyEqual(to other: Float, tolerance: Float) -> Bool {
        abs(self - other) <= tolerance
    }
}
