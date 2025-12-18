//
//  QuadBayerForwardOperatorTests.swift
//  ProjectTealTests
//
//  Synthetic harness to validate the quad-Bayer forward operator used for
//  sensor-consistency losses. Ensures mosaicing + binning mirror device logic
//  and respect per-channel scaling.

import Foundation
@testable import ProjectTeal
import Testing

private extension SIMD3 where Scalar == Float {
    init(repeating value: Float) {
        self.init(value, value, value)
    }
}

struct QuadBayerForwardOperatorTests {
    @Test func binsQuadBlocksWithRGGBPattern() throws {
        // 4×4 high-res image => 2×2 mosaiced output.
        let width = 4
        let height = 4

        // Encode channel-distinct ramps so the expected averages are easy to compute.
        var pixels = [SIMD3<Float>]()
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = Float(y * width + x)
                pixels.append(SIMD3<Float>(base, base + 100, base + 200))
            }
        }
        let rgbImage = RGBImage(width: width, height: height, pixels: pixels)

        let operatorUnderTest = QuadBayerForwardOperator()
        let mosaiced = operatorUnderTest.apply(to: rgbImage, cfaPattern: .rggb)

        #expect(mosaiced.width == 2)
        #expect(mosaiced.height == 2)

        // Expected averages per 2×2 block following RGGB quad layout.
        // Block (0,0) -> Red channel average of r0...r3.
        let expectedR = (0 + 1 + 4 + 5).asFloat * 0.25
        // Block (1,0) -> Green channel average (g values = base + 100).
        let expectedGTop = (2 + 3 + 6 + 7).asFloat * 0.25 + 100
        // Block (0,1) -> Green channel average for bottom-left block.
        let expectedGBottom = (8 + 9 + 12 + 13).asFloat * 0.25 + 100
        // Block (1,1) -> Blue channel average (b values = base + 200).
        let expectedB = (10 + 11 + 14 + 15).asFloat * 0.25 + 200

        #expect(mosaiced[0, 0].isApproximatelyEqual(to: expectedR, tolerance: 1e-6))
        #expect(mosaiced[1, 0].isApproximatelyEqual(to: expectedGTop, tolerance: 1e-6))
        #expect(mosaiced[0, 1].isApproximatelyEqual(to: expectedGBottom, tolerance: 1e-6))
        #expect(mosaiced[1, 1].isApproximatelyEqual(to: expectedB, tolerance: 1e-6))
    }

    @Test func appliesPerChannelScalingBeforeBinning() throws {
        let width = 2
        let height = 2
        let pixels = [SIMD3<Float>(repeating: 1)]
            .repeated(count: width * height)
        let rgbImage = RGBImage(width: width, height: height, pixels: pixels)

        let operatorUnderTest = QuadBayerForwardOperator()
        let scale = SIMD3<Float>(2, 3, 4)
        let mosaiced = operatorUnderTest.apply(to: rgbImage,
                                               cfaPattern: .gbrg,
                                               channelScale: scale)

        // For a 2×2 input, every pixel in the block maps to the same channel for quad Bayer.
        // The output should equal the channel's scale factor.
        #expect(mosaiced.width == 1)
        #expect(mosaiced.height == 1)
        #expect(mosaiced[0, 0].isApproximatelyEqual(to: scale[0], tolerance: 1e-6))
    }
}

private extension Float {
    var asFloat: Float { self }

    func isApproximatelyEqual(to other: Float, tolerance: Float) -> Bool {
        abs(self - other) <= tolerance
    }
}

private extension Array where Element == SIMD3<Float> {
    func repeated(count: Int) -> [SIMD3<Float>] {
        precondition(!isEmpty, "Array must not be empty to repeat")
        return (0 ..< count).map { _ in self[0] }
    }
}
