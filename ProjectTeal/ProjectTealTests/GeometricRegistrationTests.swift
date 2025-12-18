//
//  GeometricRegistrationTests.swift
//  ProjectTealTests
//
//  Synthetic harness to validate the alignment estimator used to lock
//  ProRAW48 to the 12MP RAW anchor.
//

import Foundation
@testable import ProjectTeal
import Testing

private struct AlignmentFixture: Decodable {
    let description: String
    let lowResWidth: Int
    let lowResHeight: Int
    let lowResPixels: [Float]
    let highResWidth: Int
    let highResHeight: Int
    let highResPixels: [Float]
    let expectedTranslation: [Double]
    let minScore: Double
}

extension AlignmentFixture {
    static func load(named name: String = "AlignmentFixture") throws -> AlignmentFixture {
        let fileURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("\(name).json")

        let data = try Data(contentsOf: fileURL)
        return try JSONDecoder().decode(AlignmentFixture.self, from: data)
    }
}

struct GeometricRegistrationTests {
    @Test func findsTranslationBetweenDownsampledPairs() async throws {
        // Create a 6x6 low-res pattern.
        let lowWidth = 6
        let lowHeight = 6
        var lowPixels = [Float](repeating: 0, count: lowWidth * lowHeight)
        for y in 0 ..< lowHeight {
            for x in 0 ..< lowWidth {
                lowPixels[y * lowWidth + x] = Float(x + y)
            }
        }
        let lowRes = GrayscaleImage(width: lowWidth, height: lowHeight, pixels: lowPixels)

        // Upsample to 12x12 and apply a 2-pixel translation in high-res space.
        let highWidth = lowWidth * 2
        let highHeight = lowHeight * 2
        var highPixels = [Float](repeating: 0, count: highWidth * highHeight)
        let highShift = SIMD2<Int>(2, -2)
        for y in 0 ..< lowHeight {
            for x in 0 ..< lowWidth {
                let value = lowRes[x, y]
                for dy in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        let hx = x * 2 + dx + highShift.x
                        let hy = y * 2 + dy + highShift.y
                        if hx >= 0 && hx < highWidth && hy >= 0 && hy < highHeight {
                            highPixels[hy * highWidth + hx] = value
                        }
                    }
                }
            }
        }
        let highRes = GrayscaleImage(width: highWidth, height: highHeight, pixels: highPixels)

        let registration = GeometricRegistration()
        let result = registration.estimateAlignment(highRes: highRes, lowRes: lowRes, searchRadius: 3)

        #expect(result.scale == SIMD2<Double>(repeating: 2))
        #expect(result.translation == SIMD2<Double>(1, -1))
        #expect(result.score > 0.99)
    }

    @Test func prefersNoShiftWhenImagesMatch() throws {
        let width = 4
        let height = 4
        let pixels: [Float] = [
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3
        ]
        let lowRes = GrayscaleImage(width: width, height: height, pixels: pixels)

        var highPixels = [Float](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let value = lowRes[x, y]
                for dy in 0 ..< 2 {
                    for dx in 0 ..< 2 {
                        let hx = x * 2 + dx
                        let hy = y * 2 + dy
                        highPixels[hy * width * 2 + hx] = value
                    }
                }
            }
        }
        let highRes = GrayscaleImage(width: width * 2, height: height * 2, pixels: highPixels)

        let registration = GeometricRegistration()
        let result = registration.estimateAlignment(highRes: highRes, lowRes: lowRes, searchRadius: 2)

        #expect(result.translation == SIMD2<Double>(repeating: 0))
        #expect(result.scale == SIMD2<Double>(repeating: 2))
    }

    @Test func locksToFixtureWithinThreshold() throws {
        let fixture = try AlignmentFixture.load()

        let lowRes = GrayscaleImage(width: fixture.lowResWidth,
                                    height: fixture.lowResHeight,
                                    pixels: fixture.lowResPixels)
        let highRes = GrayscaleImage(width: fixture.highResWidth,
                                     height: fixture.highResHeight,
                                     pixels: fixture.highResPixels)

        let registration = GeometricRegistration()
        let result = registration.estimateAlignment(highRes: highRes,
                                                    lowRes: lowRes,
                                                    searchRadius: 4)

        let expectedTranslation = SIMD2<Double>(fixture.expectedTranslation[0], fixture.expectedTranslation[1])
        #expect(result.translation == expectedTranslation)
        #expect(result.scale == SIMD2<Double>(repeating: 2))
        #expect(result.score >= fixture.minScore)
    }
}
