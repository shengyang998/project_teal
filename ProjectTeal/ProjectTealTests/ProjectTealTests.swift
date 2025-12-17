//
//  ProjectTealTests.swift
//  ProjectTealTests
//
//  Created by Soleil Yu on 2025/9/23.
//

import CoreGraphics
@testable import ProjectTeal
import Testing

struct ProjectTealTests {
    @Test func parsesRAW12Metadata() throws {
        let rawDictionary: [CFString: Any] = [
            kCGImagePropertyRawBlackLevel: 64.0,
            kCGImagePropertyRawWhiteLevel: 4095.0,
            kCGImagePropertyExifWhiteBalanceBias: [2.0, 1.0, 1.5],
            kCGImagePropertyDNGCFAPattern: [0, 1, 1, 2]
        ]

        let parser = RAW12Parser()
        let parsed = parser.parse(rawProperties: rawDictionary, pixelSize: CGSize(width: 4000, height: 3000))

        #expect(parsed?.blackLevel == 64)
        #expect(parsed?.whiteLevel == 4095)
        #expect(parsed?.whiteBalanceGains == SIMD3(2, 1, 1.5))
        #expect(parsed?.cfaPattern == .rggb)
        #expect(parsed?.pixelSize == CGSize(width: 4000, height: 3000))
    }

    @Test func normalizesPixelValues() {
        let normalization = RAW12Parser.Normalization(blackLevel: 64, whiteLevel: 4095, whiteBalanceGains: SIMD3(1, 1, 1))

        #expect(normalization.normalize(value: 64) == 0)
        #expect(abs(normalization.normalize(value: 2048) - 0.5) < 1e-6)
        #expect(normalization.normalize(value: 5000) == 1)
    }
}
